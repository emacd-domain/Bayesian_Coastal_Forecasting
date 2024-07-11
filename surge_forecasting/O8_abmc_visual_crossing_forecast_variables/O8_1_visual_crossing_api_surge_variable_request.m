cd ..\O1_location_selection
load domainlocs

lat=domainlocs(:,2);
lon=domainlocs(:,1);

dtseq = datetime([2020,1,1]):days(1):datetime([2022,12,31]);

cd ..\O9_abmc_visual_crossing_forecast\visual_crossing_inputs_by_year\2023

% ryr=year(dtseq(1));
% rday = day(dtseq(1)); if rday<10 rday="0"+string(rday); else rday=string(rday); end
% rmonth = month(dtseq(1)); if rmonth<10 rmonth="0"+string(rmonth); else rmonth=string(rmonth); end
% 
% eyr=year(dtseq(end));
% eday = day(dtseq(end)); if eday<10 eday="0"+string(eday); else eday=string(eday); end
% emonth = month(dtseq(end)); if emonth<10 emonth="0"+string(emonth); else emonth=string(emonth); end

for i = 1:length(domainlocs)
    i
    url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"...
          +string(lat(i))+"%2C"...
          +string(lon(i))+"/"...
          +'2023-01-01/2023-12-31?unitGroup=metric&elements=datetime%2Cwindspeed%2Cwinddir%2Cpressure&include=hours%2Cstats%2Cobs%2Cremote&key=PTBRBS9RPR3AH84936BVPUEYW&contentType=csv';
    filename = "fc_erano_"+string(i)+".csv";
    options=weboptions('Timeout',60);
    websave(filename,url,options);
end

cd ..



