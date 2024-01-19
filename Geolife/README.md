# Geolife dataset description
[Geolife docs](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/User20Guide-1.2.pdf)

## PLT format:
- Line 1…6 are useless in this dataset, and can be ignored. Points are described in following lines, one for each line.
- Field 1: Latitude in decimal degrees.
- Field 2: Longitude in decimal degrees.
- Field 3: All set to 0 for this dataset.
- Field 4: Altitude in feet (-777 if not valid).
- Field 5: Date - number of days (with fractional part) that have passed since 12/30/1899.
- Field 6: Date as a string.
- Field 7: Time as a string.

Note that field 5 and field 6&7 represent the same date/time in this dataset. You may use either of them.
Example:
39.906631,116.385564,0,492,40097.5864583333,2009-10-11,14:04:30
39.906554,116.385625,0,492,40097.5865162037,2009-10-11,14:04:35

## Transportation mode labels
Possible transportation modes are: walk, bike, bus, car, subway, train, airplane, boat, run and motorcycle. Again, we have
converted the date/time of all labels to GMT, even though most of them were created in China.

## Microsoft Research License Agreement
Non-Commercial Use Only