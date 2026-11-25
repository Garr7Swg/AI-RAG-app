# Supabase setup (auth + profile + avatars + RLS)

The app only needs the project URL and the anon/publishable key to authenticate. Everything
below makes the setup production-shaped: a profile row per user, avatars in storage, and
row-level security on everything.

## 1. Create the project

1. [supabase.com](https://supabase.com) → **New project** (free tier).
2. Copy, from **Project Settings → API**:
   - `Project URL` → `SUPABASE_URL`
   - `anon` / `publishable` key → `SUPABASE_ANON_KEY`
3. **Authentication → Providers → Email**: enabled.
4. **Authentication → URL Configuration**:
   - Site URL: your app's public origin — `https://<your-app>.streamlit.app` in the cloud,
     `http://localhost:8501` only if you exclusively work locally.
   - Redirect URLs: **every** origin you use, e.g. `http://localhost:8501` **and**
     `https://<your-app>.streamlit.app`. No wildcards.
   - This value is the fallback for confirmation links. The app normally overrides it with
     `RAG_PUBLIC_URL` (see step 6), so set both and they cannot drift.
5. For a quick demo you may disable **Confirm email**. For anything real, leave it on: the app
   handles the "please confirm your email" branch.
6. Tell the app its own public origin (Streamlit secrets on Cloud, `.env` locally):

   ```bash
   RAG_PUBLIC_URL=https://<your-app>.streamlit.app   # no trailing slash, no path
   ```

   Without it, sign-up asks Supabase to use the *Site URL*; if that still says
   `localhost`, the confirmation email sends people to a dead link on their own machine and
   the account is never confirmed — which then reads as "cannot log in".

## 2. Schema, trigger and RLS

Run this in **SQL Editor**:

```sql
-- ---------------------------------------------------------------- profiles
create table if not exists public.profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  email text not null,
  full_name text,
  avatar_url text,
  role text not null default 'analyst' check (role in ('analyst', 'admin')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.profiles enable row level security;

create policy "profiles: read own"
  on public.profiles for select
  using (auth.uid() = id);

create policy "profiles: update own"
  on public.profiles for update
  using (auth.uid() = id)
  with check (auth.uid() = id);

-- users may not change their own role
create or replace function public.protect_role()
returns trigger language plpgsql as $$
begin
  if new.role <> old.role and current_setting('request.jwt.claim.role', true) <> 'service_role' then
    new.role := old.role;
  end if;
  new.updated_at := now();
  return new;
end $$;

drop trigger if exists protect_role on public.profiles;
create trigger protect_role before update on public.profiles
  for each row execute function public.protect_role();

-- create a profile automatically on sign-up
create or replace function public.handle_new_user()
returns trigger language plpgsql security definer set search_path = public as $$
begin
  insert into public.profiles (id, email, full_name, avatar_url)
  values (
    new.id,
    new.email,
    coalesce(new.raw_user_meta_data ->> 'full_name', split_part(new.email, '@', 1)),
    new.raw_user_meta_data ->> 'avatar_url'
  )
  on conflict (id) do nothing;
  return new;
end $$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- ------------------------------------------------------------ conversations
create table if not exists public.conversations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  title text not null default 'New conversation',
  created_at timestamptz not null default now()
);

create table if not exists public.messages (
  id bigint generated always as identity primary key,
  conversation_id uuid not null references public.conversations (id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  citations jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now()
);

alter table public.conversations enable row level security;
alter table public.messages enable row level security;

create policy "conversations: owner all"
  on public.conversations for all
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

create policy "messages: owner read"
  on public.messages for select
  using (exists (
    select 1 from public.conversations c
    where c.id = messages.conversation_id and c.user_id = auth.uid()
  ));

create policy "messages: owner insert"
  on public.messages for insert
  with check (exists (
    select 1 from public.conversations c
    where c.id = messages.conversation_id and c.user_id = auth.uid()
  ));

create index if not exists messages_conversation_idx on public.messages (conversation_id, created_at);
create index if not exists conversations_user_idx on public.conversations (user_id, created_at desc);

-- --------------------------------------------------------------- documents
-- Per-user corpora are a natural extension; the RAG metadata already carries doc_id.
create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  owner_id uuid not null references auth.users (id) on delete cascade,
  doc_id text not null,
  file_name text not null,
  sha256 text not null,
  page_count int not null default 0,
  chunk_count int not null default 0,
  created_at timestamptz not null default now(),
  unique (owner_id, sha256)
);

alter table public.documents enable row level security;

create policy "documents: owner all"
  on public.documents for all
  using (auth.uid() = owner_id)
  with check (auth.uid() = owner_id);
```

## 3. Avatar storage

```sql
insert into storage.buckets (id, name, public)
values ('avatars', 'avatars', true)
on conflict (id) do nothing;

create policy "avatars: public read"
  on storage.objects for select
  using (bucket_id = 'avatars');

create policy "avatars: owner write"
  on storage.objects for insert
  with check (bucket_id = 'avatars' and auth.uid()::text = (storage.foldername(name))[1]);

create policy "avatars: owner update"
  on storage.objects for update
  using (bucket_id = 'avatars' and auth.uid()::text = (storage.foldername(name))[1]);
```

Upload path convention: `avatars/<user-id>/avatar.png`. Then store the public URL in
`user_metadata.avatar_url` (or `profiles.avatar_url`). The app only renders avatar URLs that are
https and on a known host (`supabase.co`, `gravatar.com`, `githubusercontent.com`,
`googleusercontent.com`), and rejects SVG/HTML — an attacker-supplied URL cannot become XSS.

## 4. Admin role

The app derives the admin role from verified claims, not from the client:

```bash
RAG_ADMIN_EMAILS=you@example.com,ops@example.com   # .env / Streamlit secrets / GitHub secret
```

Admins can re-index the corpus and delete documents. Analysts can chat, upload and download.

## 5. Verify it works

```bash
python scripts/check_env.py        # "Supabase configured (https://...)"
```

Then in the app: sign up → confirm (if enabled) → sign in → sidebar shows your name, initials or
avatar, and role → **Sign out** clears the session.

If sign-in fails, the UI now names the configuration problem when Supabase reports one
(e-mail rate limit, sign-ups disabled, rejected anon key, unreachable/paused project). Anything
else deliberately stays generic. Otherwise check, in this order:

1. Email confirmed? (Supabase → Authentication → Users → the user shows *Confirmed*.)
2. `RAG_PUBLIC_URL` and the Supabase *Site URL* both point at the origin you are actually
   using, and that origin is in *Redirect URLs* (scheme, host and port must match exactly).
3. The app really received the credentials: **System → Configuration** shows
   `auth_enabled: true` and the `auth_redirect_url` in use.
4. Did you paste the **anon** key, not the service-role key?
5. Is the project paused? Free projects pause after inactivity — this is a common "it worked
   yesterday" cause.
6. Confidential email confirmations are throttled on the free tier (a few per hour).
   Supabase returns *email rate limit exceeded*; the UI surfaces that verbatim.
