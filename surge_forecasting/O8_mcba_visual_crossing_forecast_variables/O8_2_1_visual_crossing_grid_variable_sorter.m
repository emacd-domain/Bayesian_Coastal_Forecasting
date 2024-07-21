% this script will read through visual crossing downloads and convert wind
% speed and direction to U10 and V10 and pressure into Pa

% change year directory in row 11 and year variable names in row 89 to 95 

load ..\O1_location_selection\domainlocs.mat

lon = domainlocs(:,1);
lat = domainlocs(:,2);

cd O8_1_visual_crossing_inputs_by_year\nowcast\2020

year_days = 366;

fcu10=nan(year_days*24,434);
fcv10=nan(year_days*24,434);
fcslp=nan(year_days*24,434);

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

cd ..\..\O8_2_visual_crossing_inputs_processed_to_era5_variables

nc_u10_2020 = fcu10;
nc_v10_2020 = fcv10;
nc_mslp_2020 = fcslp;

save("nc_mslp_2020.mat","nc_mslp_2020")
save("nc_u10_2020.mat","nc_u10_2020")
save("nc_v10_2020.mat","nc_v10_2020")

cd ..\O8_1_visual_crossing_inputs_by_year\forecast\2020

fcu10=nan(year_days*24,434);
fcv10=nan(year_days*24,434);
fcslp=nan(year_days*24,434);

for j=2:year_days*24
    j
    foldername="daynumber"+string(j);
    cd (foldername)
    for i=1:434
        filename="fc_erano_"+string(i)+".csv";
        fcast=readtable(filename);

        if height(fcast)==0
            continue
        else
            if height(fcast)>24
                [~,r]=unique(table2array(fcast(:,'datetime')));
                fcast=fcast(r,:);
            end
    
            if height(fcast)<24
                fcast = timetable(table2array(fcast(:,'datetime')),table2array(fcast(:,'windspeed')),...
                                table2array(fcast(:,'winddir')));%,table2array(fcast(:,'sealevelpressure')));
                fcast = retime(fcast,'hourly');
                fcast = timetable2table(fcast);
                fcast.Properties.VariableNames = ["datetime","windspeed","winddir"];%,"sealevelpressure"];
            end
    
            windspeed = 0.278*table2array(fcast(:,'windspeed'));
            winddir = table2array(fcast(:,'winddir'));
            winddir = 270 - winddir;
            u10 = windspeed.*cosd(winddir);
            v10 = windspeed.*sind(winddir);
            %pressure = 100*table2array(fcast(:,'sealevelpressure'));
            
            p = (j*24)-23;
            if ~isempty(u10)
                fcu10(p:p+23,i)=u10;
            end
            if ~isempty(v10)
                fcv10(p:p+23,i)=v10;
            end
            if ~isempty(pressure)
                fcslp(p:p+23,i)=pressure;
            end
        end
    end
    cd ..
end

for i=1:year_days*24
    i
    indexn=find(isnan(fcu10(i,:)));
    indexp=find(~isnan(fcu10(i,:)));
    if ~isempty(indexn)
         fcu10(i,indexn)=griddata(lon(indexp),lat(indexp),fcu10(i,indexp),lon(indexn),lat(indexn),'nearest');
    end
     
     indexn=find(isnan(fcv10(i,:)));
     indexp=find(~isnan(fcv10(i,:)));
     if ~isempty(indexn)
         fcv10(i,indexn)=griddata(lon(indexp),lat(indexp),fcv10(i,indexp),lon(indexn),lat(indexn),'nearest');
     end
   
   indexn=find(isnan(mslp(i,:)));
   indexp=find(~isnan(mslp(i,:)));
   if ~isempty(indexn)
       msl = scatteredInterpolant(lon(indexp),lat(indexp),mslp(i,indexp)');
       mslpq=msl(lon(indexn),lat(indexn));%%,mslp(i,indexp),lon(indexn),lat(indexn),'nearest');
       mslpq(mslpq<min(mslp(i,indexp)))=min(mslp(i,indexp));
       mslpq(mslpq>max(mslp(i,indexp)))=max(mslp(i,indexp));
       mslpq(mslpq==0)=mean(mslp(i,indexp));
       fcslp(i,indexn)=mslpq;
   end
end

cd ..\..\..\O8_2_visual_crossing_inputs_processed_to_era5_variables

fc_u10_2020 = fcu10;
fc_v10_2020 = fcv10;
fc_mslp_2020 = fcslp;

save("fc_mslp_2020.mat","fc_mslp_2020")
save("fc_u10_2020.mat","fc_u10_2020")
save("fc_v10_2020.mat","fc_v10_2020")