load ..\O1_location_selection\domainlocs

cd O8_1_visual_crossing_inputs_by_year\forecast\2020
dtseq = datetime([2020,1,1]):days(1):datetime([2020,12,31]);

key = "PTBRBS9RPR3AH84936BVPUEYW";

lat=domainlocs(:,2);
lon=domainlocs(:,1);

for j = 2:length(dtseq)
    disp(string(j))  
    system("mkdir daynumber"+string(j));
    cd ("daynumber"+string(j))
    fyr = year(dtseq(j-1));
    fday = day(dtseq(j-1)); if fday<10 fday="0"+string(fday); else fday=string(fday); end
    fmonth = month(dtseq(j-1)); if fmonth<10 fmonth="0"+string(fmonth); else fmonth=string(fmonth); end        
    ryr=year(dtseq(j));
    rday = day(dtseq(j)); if rday<10 rday="0"+string(rday); else rday=string(rday); end
    rmonth = month(dtseq(j)); if rmonth<10 rmonth="0"+string(rmonth); else rmonth=string(rmonth); end
    for i = 1:length(domainlocs)              
        url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"...
              +string(lat(i))+"%2C"...
              +string(lon(i))+"/"...
              +ryr+"-"+rmonth+"-"+rday+"/"...
              +ryr+"-"+rmonth+"-"+rday...
              +"?unitGroup=metric&elements=datetime%2Cwindspeed%2Cwinddir%2Cpressure&include=hours%2Cobs%2Cremote%2Cstats%2Cstatsfcst%2Cfcst&key="+key+"&contentType=csv&"...
              +"forecastBasisDate="+fyr+"-"+fmonth+"-"+fday;
        filename = "fc_erano_"+string(i)+".csv";
        options=weboptions('Timeout',60);
        websave(filename,url,options);
    end
    cd ..
end

cd ..\..\nowcast\2020

ryr=year(dtseq(1));
rday = day(dtseq(1)); if rday<10 rday="0"+string(rday); else rday=string(rday); end
rmonth = month(dtseq(1)); if rmonth<10 rmonth="0"+string(rmonth); else rmonth=string(rmonth); end

eyr=year(dtseq(end));
eday = day(dtseq(end)); if eday<10 eday="0"+string(eday); else eday=string(eday); end
emonth = month(dtseq(end)); if emonth<10 emonth="0"+string(emonth); else emonth=string(emonth); end

for i = 1:length(domainlocs)
    i
    url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"...
          +string(lat(i))+"%2C"...
          +string(lon(i))+"/"...
          +'2022-01-01/2022-12-31?unitGroup=metric&elements=datetime%2Cwindspeed%2Cwinddir%2Cpressure&include=hours%2Cstats%2Cobs%2Cremote&key='+key+'&contentType=csv';
    filename = "fc_erano_"+string(i)+".csv";
    options=weboptions('Timeout',60);
    websave(filename,url,options);
end

cd ..\..



