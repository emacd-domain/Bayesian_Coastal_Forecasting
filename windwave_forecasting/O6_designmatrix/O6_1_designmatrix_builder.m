load ..\O3_cmems_wave_model\cmems_wavechar_model.mat
load ..\O5_pca\era5_pcs_1980_2019.mat
load ..\O5_pca\era5_tt_u10_1980_2019.mat
load ..\O5_pca\era5_tt_v10_1980_2019.mat

% sync timetables for features and targets
tt_master = synchronize(era5_pcs_1980_2019, era5_tt_u10_1980_2019, era5_tt_v10_1980_2019, cmems_wavechar_model);

% get directional components from wave from direction
wave_from = tt_master.mdr;
D = 270-wave_from;
U = cosd(D);
V = sind(D);

% create array from sync timetable
dm = [tt_master.era5_pcs_1980_2019,tt_master.U, tt_master.V, tt_master.hm0, tt_master.tpk, U, V];

index_m = find(dm(:,45) == 0);
dm(index_m, 45:end) = nan;

% create 3D array [observations, features, timesteps]
s=size(dm);

% fill 3D array
dm_3D = zeros([s(1), s(2), 25]);
for i=1:1:length(dm)-24
    i
    dm_3D(i,:,:) = dm(i:i+24,:)';
end

dt_series = (datetime([1980,1,1,0,0,0]):hours(1):datetime([2019,12,31,23,0,0]))';
idx_1 = find(dt_series == datetime([2000, 1, 1, 0, 0, 0]));
idx_2 = find(dt_series == datetime([2001, 12, 31, 23, 0, 0]));

dm_3D_rem = dm_3D(idx_1:idx_2, :, :);
dm_3D(idx_1:idx_2, :, :) = [];

% normalise designmatrix
dm_mean = nanmean(dm_3D, [1,3]);
dm_std = nanstd(dm_3D, [], [1,3]);
dm_normalised = (dm_3D - repmat(dm_mean,[length(dm_3D),1]))./repmat(dm_std,[length(dm_3D),1]);
dm_rem_normalised = (dm_3D_rem - repmat(dm_mean,[length(dm_3D_rem),1]))./repmat(dm_std,[length(dm_3D_rem),1]);

% remove nan
index = sum(isnan(dm_normalised), [2,3]) == 0;
dm_normalised = dm_normalised(index, :, :);

% save 
save('dm_normalised.mat', "dm_normalised",'-v7.3')
save('dm_rem_normalised.mat', "dm_rem_normalised")
save('dm_mean.mat', "dm_mean")
save('dm_std.mat', "dm_std")