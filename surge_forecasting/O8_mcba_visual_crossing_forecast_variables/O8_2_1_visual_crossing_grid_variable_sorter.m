% this script will read through visual crossing downloads and convert wind
% speed and direction to U10 and V10 and pressure into Pa

% change year directory in row 11 and year variable names in row 89 to 95 

load ..\O1_location_selection\domainlocs.mat

lon = domainlocs(:,1);
lat = domainlocs(:,2);

cd O8_1_visual_crossing_inputs_by_year\2020

fcu10=nan(366*24,434);
fcv10=nan(366*24,434);
fcslp=nan(366*24,434);

for i=1:434
    i
    filename="fc_erano_"+string(i)+".csv";
    fcast=readtable(filename);

    if height(fcast)==0
        continue
    else
        if height(fcast)>365*24
            [~,r]=unique(table2array(fcast(:,'datetime')));
        fcast=fcast(r,:);
        end
        
        if height(fcast)<365*24
            fcast = timetable(table2array(fcast(:,'datetime')),table2array(fcast(:,'windspeed')),...
                              table2array(fcast(:,'winddir')),table2array(fcast(:,'sealevelpressure')));
            fcast = retime(fcast,'hourly');
            fcast = timetable2table(fcast);
            fcast.Properties.VariableNames = ["datetime","windspeed","winddir","sealevelpressure"];
        end
        fcast=table2timetable(fcast);
        fcast=retime(fcast,'hourly');
        windspeed = 0.278*table2array(fcast(:,'windspeed'));
        winddir = table2array(fcast(:,'winddir'));
        winddir = 270 - winddir;
        u10 = windspeed.*cosd(winddir);
        v10 = windspeed.*sind(winddir);
        pressure = 100*table2array(fcast(:,'sealevelpressure'));
                
        
        if ~isempty(u10)
            fcu10(:,i)=u10;
        end
        if ~isempty(v10)
            fcv10(:,i)=v10;
        end
        if ~isempty(pressure)
            fcslp(:,i)=pressure;
        end
    end
end

cd ..\..\O8_2_visual_crossing_inputs_processed_to_era5_variables

for i=1:8784
    i
    indexn=find(isnan(fcu10(i,:)));
    indexp=find(~isnan(fcu10(i,:)));
    if ~isempty(indexn)
         fcu10(i,indexn)=griddata(lon(indexp),lat(indexp),fcu10(i,indexp),lon(indexn),lat(indexn),'nearest');
         fcv10(i,indexn)=griddata(lon(indexp),lat(indexp),fcv10(i,indexp),lon(indexn),lat(indexn),'nearest');
    end
     
     indexn=find(isnan(fcu10(i,:)));
     indexp=find(~isnan(fcu10(i,:)));
     if ~isempty(indexn)
         fcv10(i,indexn)=griddata(lon(indexp),lat(indexp),fcv10(i,indexp),lon(indexn),lat(indexn),'nearest');
         fcu10(i,indexn)=griddata(lon(indexp),lat(indexp),fcu10(i,indexp),lon(indexn),lat(indexn),'nearest');
     end
    
    indexn=find(isnan(fcslp(i,:)));
    indexp=find(~isnan(fcslp(i,:)));
    if ~isempty(indexn)
        msl = scatteredInterpolant(lon(indexp),lat(indexp),fcslp(i,indexp)');
        mslpq=msl(lon(indexn),lat(indexn));
        mslpq(mslpq<min(fcslp(i,indexp)))=min(fcslp(i,indexp));
        mslpq(mslpq>max(fcslp(i,indexp)))=max(fcslp(i,indexp));
        mslpq(mslpq==0)=mean(fcslp(i,indexp));
        fcslp(i,indexn)=mslpq;
    end
end

vc_u10_2020 = fcu10;
vc_v10_2020 = fcv10;
vc_mslp_2020 = fcslp;

save("vc_mslp_2020.mat","vc_mslp_2020")
save("vc_u10_2020.mat","vc_u10_2020")
save("vc_v10_2020.mat","vc_v10_2020")

