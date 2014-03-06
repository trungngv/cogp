This data contains weather information collected by sensors at 4 different locations in England.
It is retrieved for the period from 01/07/2013 - 15/07/2013 from the websites
http://www.cambermet.co.uk/
http://www.bramblemet.co.uk/
http://www.sotonmet.co.uk/
http://www.chimet.co.uk/

The abbreviations for the measures are:

WSPD	 Wind Speed in knots
WD	 Wind Direction in degrees
GST	 Maximum Gust in knots
ATMP	 Air Temperature in degrees C
WTMP	 Water Temperature in degrees C
BARO	 Barometric Pressure in millibars
DEPTH	 Water depth in metres
AWVHT	 Average Wave Height in metres
MWVHT	 Maximum Wave Height in metres
WVHT	 Significant Wave Height in metres
APD	 Average Wave Period in seconds

The websites provide daily measures in a single file so after collecting 15 files 
for the given period, I concatenate them into the 4 files corresponding to the 4
locations: bra.csv, cam.csv, chi.csv, sot.csv.

Since different measurements are collected at different locations, I create new csv files
new{xxx}.csv where only  WSPD,WD,GST, ATMP are retained.

After processing, the remaining fields are
Date,Hour,Minute,WSPD,WD,GST,ATMP
where date = {1,..,15}, hour = {00,...,23}, minute = {00,05,...,55}
if any combination of (date,hour,minute) does not exist, the data is missing.

Missing data:
bra : 100
cam: 0
chi: 15
sot: 1002

