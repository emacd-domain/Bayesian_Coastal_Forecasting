load ..\O4_millport_surge\ntslf_tt_millport_surge.mat
load ..\O3_pca\era5_pcs_1980_2019.mat
load ..\O3_pca\era5_tt_msl_pressure.mat
load ..\O3_pca\era5_tt_u10.mat
load ..\O3_pca\era5_tt_v10.mat

% sync timetables for features and targets
tt_master = synchronize(era5_pcs_1980_2019, ntslf_tt_millport_surge, era5_tt_msl_pressure, era5_tt_u10, era5_tt_v10);

% create array from sync timetable
dm = [tt_master.era5_pcs_95, tt_master.M, tt_master.U, tt_master.V, tt_master.residual];

% create 3D array [observations, features, timesteps]
s=size(dm);

% fill 3D array
dm_3D = zeros([s(1), s(2), 49]);
for i=1:1:length(dm)-48
    i
    dm_3D(i,:,:) = dm(i:i+48,:)';
end

% normalise designmatrix
dm_mean = nanmean(dm_3D, [1,3]);
dm_std = nanstd(dm_3D, [], [1,3]);
dm_normalised = (dm_3D - repmat(dm_mean,[length(dm_3D),1]))./repmat(dm_std,[length(dm_3D),1]);

% remove nan
index = sum(isnan(dm_normalised), [2,3]) == 0;
dm_normalised = dm_normalised(index, :, :);

% save 
save('dm_normalised.mat', "dm_normalised")
save('dm_mean.mat', "dm_mean")
save('dm_std.mat', "dm_std")