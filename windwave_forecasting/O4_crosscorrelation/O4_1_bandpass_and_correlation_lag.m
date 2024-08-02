% this file will extract the points from netcdffiles, filter them based
% on the fetch radius and retime them
load ..\O1_location_selection\domainlocs.mat

currentFolder = pwd

clearvars -except domainlocs

%load pointlatlon
windlon=domainlocs(:,1);
windlat=domainlocs(:,2);

% get u10 filenames extracted at domain locations
filenames_u10=[];
dinfo = dir('..\O2_era5_variables\u10_wind_at_domain_locs\*.mat');
for K=1:length(dinfo)
  iname = string(dinfo(K).name);
  filenames_u10=[filenames_u10;iname];
end

% get v10 filenames extracted at domain locations
filenames_v10=[];
dinfo = dir('..\O2_era5_variables\v10_wind_at_domain_locs\*.mat');
for K=1:length(dinfo)
  iname = string(dinfo(K).name);
  filenames_v10=[filenames_v10;iname];
end

% load U10
for i=1:length(filenames_u10)
    u10 = load('..\O2_era5_variables\u10_wind_at_domain_locs\'+filenames_u10(i),"-mat");
    u10 = u10.U10;
    
    if (i == 1)
        U = u10;
    else
        U = [U; u10];
    end
end

% load V10
for i=1:length(filenames_u10)
    v10 = load('..\O2_era5_variables\v10_wind_at_domain_locs\'+filenames_v10(i),"-mat");
    v10 = v10.V10;
    
    if (i == 1)
        V = v10;
    else
        V = [V; v10];
    end
end

% get wave char
load '..\O3_cmems_wave_model\cmems_wavechar_model.mat'
Hm0 = cmems_wavechar_model.hm0(find(cmems_wavechar_model.dtr == datetime(2000,1,1)):find(cmems_wavechar_model.dtr == datetime(2004,12,31,23,0,0)),:);

Hm0_filter = highpass(Hm0, [1/(24*14)], 1);

max_correlation_lags = zeros([length(windlon),1]);

for i=258%1:length(domainlocs)
    i
    % Calculate the Magnitudes and Directions from U10 and V10
    Magnitude = sqrt(U.u10ord(:, i).^2+V.v10ord(:, i).^2);
    Theta = atand((-U.u10ord(:, i))./V.v10ord(:, i));
    % get angle from north between points
    dX = -5 - windlon(i);
    dY = 55.75 - windlat(i);
    loc_to_target_Theta = atan2d(dX,dY)+180;
    % set modular min and max
    dir_min = mod(loc_to_target_Theta - 45, 360);
    dir_max = mod(loc_to_target_Theta + 45, 360);
    if dir_min > dir_max
        index = find((Theta>=dir_min)&(Theta<=dir_max));
    else
        index = find((Theta>=dir_min)|(Theta<=dir_max));
    end
    % mask and bandpass Magnitudes and Find Cross Correlation with Significant Wave Height
    
    Magnitude(~index) = 0;
    Hm0(~index) = 0;

    %Magnitude_filter = highpass(Magnitude, 1/(24*14), 1);
    [C, LAGS] = xcorr(Magnitude, Hm0, 36,'normalized');

    %stem(LAGS, C)
    %C(38:end) = 0;
    [M,I] = max(C);
    max_correlation_lags(i) = LAGS(I);
end

save("max_corelation_lags.mat","max_correlation_lags")