# CX-900 Centrifugal Chiller — Installation, Operation and Maintenance Manual

Document 47-1190, revision F. Applies to CX-900 units with controller firmware **3.8.2** or later.

## 1. Specifications

| Parameter | Value |
| --- | --- |
| Nominal cooling capacity | 900 tons (3,165 kW) |
| Refrigerant | R-513A |
| Refrigerant charge | 480 kg per circuit |
| Oil type | Polyolester (POE), ISO 68 viscosity grade |
| Oil charge | 38 L per circuit |
| Electrical supply | 460 V, 3-phase, 60 Hz |
| Maximum input power | 640 kW |
| Evaporator design flow | 545 m³/h |
| Condenser design flow | 655 m³/h |
| Full-load efficiency | 0.52 kW/ton at AHRI 550/590 conditions |
| Maximum condenser water temperature | 38 °C |
| Vibration limit | 4.5 mm/s RMS at the compressor bearing housing |
| Controller protocol | Modbus RTU and BACnet MS/TP |
| Warranty | 24 months from commissioning or 30 months from shipment, whichever is earlier |

R-513A is a low-pressure refrigerant with zero ozone-depletion potential. The CX-900 uses a single
centrifugal compressor per circuit and a falling-film evaporator.

## 2. Safety

Installation and service must follow OSHA 29 CFR 1910.147 for lockout/tagout. Isolate and lock the
main disconnect before opening any electrical panel. Refrigerant handling requires EPA Section 608
certification. Wear safety glasses and cryogenic gloves when working on refrigerant circuits.
Never weld or braze on a circuit that has not been evacuated to 500 microns. The purge unit must
remain energised whenever the chiller contains refrigerant.

Do not operate the compressor with the evaporator water flow below the design minimum. Do not
disable the high-pressure cutout; it is a safety device, not a control input.

## 3. Pre-Startup Checklist

1. Energise the oil heater at least 24 hours before the first start.
2. Confirm compressor oil level is between the sight-glass marks and oil temperature is at least 40 °C.
3. Verify evaporator and condenser water loops are filled, vented and chemically treated.
4. Confirm water quality: pH between 7.5 and 9.0, chloride below 200 ppm, conductivity below
   3,000 µS/cm.
5. Check that the main disconnect is closed and control voltage is 24 V DC ±10%.
6. Record baseline vibration and bearing temperatures.
7. Download and archive the controller configuration before changing any setpoint.

## 4. Scheduled Maintenance

| Task | Interval |
| --- | --- |
| Oil change (POE ISO 68, 38 L) | 4,000 operating hours or 12 months, whichever comes first |
| Oil filter replacement | With every oil change |
| Bearing inspection | Every 8,000 operating hours |
| Refrigerant leak check | Every 12 months |
| Condenser tube cleaning | Annually, or when approach temperature exceeds alarm threshold |
| Controller firmware review | Every 6 months |

When reinstalling the bearing cap bolts, torque them to **45 N·m** in a star pattern in two passes.
Replace the cap gasket whenever the cap is removed. Do not reuse a compressed gasket.

## 5. Alarms and Troubleshooting

| Code | Description | First action |
| --- | --- | --- |
| E-12 | Low evaporator pressure | Verify evaporator water flow and refrigerant charge |
| E-27 | High discharge temperature | Inspect condenser water flow and fouling; confirm charge |
| E-41 | Condenser water flow loss | Check pump status, strainer and differential pressure switch |
| E-58 | Sensor drift detected | Recalibrate or replace the affected temperature sensor |

A condenser approach temperature above **4.2 K** triggers a maintenance alarm; it usually indicates
fouling or non-condensable gas in the circuit. Clear an alarm only after the underlying condition is
corrected. Resetting an alarm without diagnosis voids the warranty for the affected component.

## 6. Efficiency and Part-Load Operation

The CX-900 achieves 0.52 kW/ton at full load under AHRI 550/590 conditions. Part-load performance
relies on variable guide vanes and a variable-frequency drive; below 25% load, the controller may
cycle the compressor to protect the motor. For data-centre duty, enable the chilled-water reset
schedule to improve integrated part-load value by 6% to 9%.

## 7. Warranty and Service

The standard warranty is 24 months from commissioning or 30 months from shipment, whichever is
earlier. Warranty coverage requires documented maintenance at the intervals in Section 4, use of
genuine Northwind filters, and refrigerant records that satisfy EPA reporting requirements. Extended
warranty of 60 months is available when a Northwind service agreement is purchased at commissioning.
