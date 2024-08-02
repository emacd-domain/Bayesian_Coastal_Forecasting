% replace with the index of the local variable
req = 666;

load ..\O1_location_selection\domainlocs.mat

% get filenames for u10
filenames_u10=[];
dinfo = dir('..\O2_era5_variables\u10_wind_at_domain_locs\*.mat');
for K=1:length(dinfo)
  iname = string(dinfo(K).name);
  filenames_u10=[filenames_u10;iname];
end

% load files and concatenate u10
for i=1:length(filenames_u10)
    u10 = load('..\O2_era5_variables\u10_wind_at_domain_locs\'+filenames_u10(i),"-mat");
    u10 = u10.U10;
    time = u10.dtime;

    if (i == 1)
        U = u10.u10ord(:,req);
        T = time;
    else
        U = [U; u10.u10ord(:,req)];
        T = [T; time];
    end
end

era5_tt_u10 = timetable(U,'RowTimes',T);
era5_tt_u10 = retime(era5_tt_u10,'hourly');

% save residual timetable
save('era5_tt_u10.mat', "era5_tt_u10")

% get filenames for v10
filenames_v10=[];
dinfo = dir('..\O2_era5_variables\v10_wind_at_domain_locs\*.mat');
for K=1:length(dinfo)
  iname = string(dinfo(K).name);
  filenames_v10=[filenames_v10;iname];
end

% load files and concatenate v10
for i=1:length(filenames_v10)
    v10 = load('..\O2_era5_variables\v10_wind_at_domain_locs\'+filenames_v10(i),"-mat");
    v10 = v10.V10;
    time = v10.dtime;
    
    if (i == 1)
        V = v10.v10ord(:,req);
        T = time;
    else
        V = [V; v10.v10ord(:,req)];
        T = [T; time];
    end
end

era5_tt_v10 = timetable(V,'RowTimes',T);
era5_tt_v10 = retime(era5_tt_v10,'hourly');

% save residual timetable
save('era5_tt_v10.mat', "era5_tt_v10")
