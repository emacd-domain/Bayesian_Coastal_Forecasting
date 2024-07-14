% this script will read Firth of CLyde local pressure in Pa

cd O8_1_visual_crossing_inputs_by_year\2023

req = 362;

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
            vc_u10(:,i)=u10;
        end
        if ~isempty(v10)
            vc_v10(:,i)=v10;
        end
        if ~isempty(pressure)
            vc_mslp(:,i)=pressure;
        end
    end
end

vc_mslp_foc_2023 = max(vc_mslp, [], 2) - vc_mslp(:, req);
vc_u10_foc_2023 = vc_u10(:, req);
vc_v10_foc_2023 = vc_v10(:, req);

cd ..\..\O8_2_visual_crossing_inputs_processed_to_era5_variables

save("vc_mslp_foc_2023.mat","vc_mslp_foc_2023")
save("vc_u10_foc_2023.mat","vc_u10_foc_2023")
save("vc_v10_foc_2023.mat","vc_v10_foc_2023")