%%%%%%%%%%%%%%%%
%%% FORECAST %%% 
%%%%%%%%%%%%%%%%

clear
clc

st=[];bi=[];in=[];se=[];pr=[];p=[];

% import era5 principal components
load ..\O3_pca\era5_pcs_2020.mat
era5_pcs_2020 = era5_pcs_2020.era5_pcs_95;
% import visual crossing principal components
load O8_3_visual_crossing_pcs\fc_pcs_2020.mat
% import design matrix mean and standard deviations
load ..\O5_designmatrix\dm_mean.mat
load ..\O5_designmatrix\dm_std.mat

s=size(era5_pcs_2020);

% normalise principal components using mean and standard deviations
norm_fc_pcs_2020 = (fc_pcs_2020 - repmat(dm_mean(1:18), [s(1), 1]))./repmat(dm_std(1:18), [s(1), 1]);
norm_era5_pcs_2020 = (era5_pcs_2020 - repmat(dm_mean(1:18), [s(1), 1]))./repmat(dm_std(1:18), [s(1), 1]);

% make adj pcs array
norm_adj_fc_pcs_2020 = zeros(s);

for i=1:s(2)
    
    idx = find(abs(norm_fc_pcs_2020(:,i) - norm_era5_pcs_2020(:, i)) < 5); 
    mdl=fitlm(norm_fc_pcs_2020(idx,i), norm_era5_pcs_2020(idx, i));
    fc_adj = table2array(mdl.Coefficients(2,1))*norm_fc_pcs_2020(:,i)+table2array(mdl.Coefficients(1,1));
    
    error=norm_era5_pcs_2020(:, i)-fc_adj;
    x=1;
    for j =0:10:90
        p1=prctile(norm_era5_pcs_2020(:, i),j);
        p2=prctile(norm_era5_pcs_2020(:, i),j+10);
        ind = find((norm_era5_pcs_2020(:, i)>=p1)&(norm_era5_pcs_2020(:, i)<p2));
        pt = j;
        pte = j+10;
        pd = fitdist(error(ind),'Normal');
        se(x,i)=pd.sigma;
        me(x,i)=pd.mu;
        pb(x,i)=p1;
        
        if x == 10
            pb(11)=p2;
        end
        x=x+1;
    %p(i)=pt;
    end
    %se(:,i)=max(se(:,i));
    st(i)=nanstd(error, 0);
    in(i)=table2array(mdl.Coefficients(1,1));
    bi(i)=table2array(mdl.Coefficients(2,1));
    norm_adj_fc_pcs_2020(:,i)=fc_adj;

end

pc_error=st';
pc_slopes=bi';
pc_intercepts=in';
pc_fc_binned_error = se;
pc_fc_binned_mean = me;
pc_fc_binned_error_thresholds = pb;
pc_fc_binned_error_thresholds(1,:) = -inf;
pc_fc_binned_error_thresholds(end,:) = inf;

% change labels between u10 and v10
ax = axes(); 
hold on
h(1)=plot(norm_fc_pcs_2020(:, 1), norm_era5_pcs_2020(:, 1),'.b','DisplayName', "Before Correction");
set(gca,'fontname','times','fontsize',8)
ax=gca;
ax.FontSize = 12;
h(2)=plot(norm_adj_fc_pcs_2020(:, 1), norm_era5_pcs_2020(:, 1),'.k','DisplayName',"After Correction");
h(3)=plot([-7, 7],[-7, 7],':r', LineWidth=2, DisplayName="y = x");
grid on
%legend(,,"y=x", FontSize=12, )
%title(["Visual Crossing Bias Correction"], FontSize=14)
xlabel(["Visual Crossing", "Normalised Principal Component 1", ], FontSize=14)
ylabel(["ERA-5","Normalised Principal Component 1"], FontSize=14)

hCopy = copyobj(h, ax); 
% replace coordinates with NaN 
% Either all XData or all YData or both should be NaN.
set(hCopy(1),'XData', NaN', 'YData', NaN)
set(hCopy(2),'XData', NaN', 'YData', NaN)
% Note, these lines can be combined: set(hCopy,'XData', NaN', 'YData', NaN)
% To avoid "Data lengths must match" warning, assuming hCopy is a handle array, 
% use arrayfun(@(h)set(h,'XData',nan(size(h.XData))),hCopy)
% Alter the graphics properties
hCopy(1).MarkerSize = 12; 
hCopy(1).LineWidth = 2;
hCopy(2).MarkerSize = 12; 
hCopy(2).LineWidth = 2; 
hCopy(3).LineWidth = 2; 
% Create legend 
legend(hCopy,Location='northwest')
xlim([-7,7])
ylim([-7,7])

cd O8_4_visual_crossing_pc_bias_correction

%save("pc_slopes.mat", "pc_slopes")
%save("pc_intercepts.mat", "pc_intercepts")
%save("pc_error.mat","pc_error")
save("pc_fc_binned_error.mat","pc_fc_binned_error")
save("pc_fc_binned_mean.mat","pc_fc_binned_mean")
save("pc_fc_binned_error_thresholds.mat", "pc_fc_binned_error_thresholds")


clearvars -except era5_pcs_2020 dm_mean dm_std
clc

%%%%%%%%%%%%%%%
%%% NOWCAST %%% 
%%%%%%%%%%%%%%%

st=[];bi=[];in=[];se=[];pr=[];p=[];

% import visual crossing principal components
load ..\O8_3_visual_crossing_pcs\nc_pcs_2020.mat

s=size(era5_pcs_2020);

% normalise principal components using mean and standard deviations
norm_nc_pcs_2020 = (nc_pcs_2020 - repmat(dm_mean(1:18), [s(1), 1]))./repmat(dm_std(1:18), [s(1), 1]);
norm_era5_pcs_2020 = (era5_pcs_2020 - repmat(dm_mean(1:18), [s(1), 1]))./repmat(dm_std(1:18), [s(1), 1]);

% make adj pcs array
norm_adj_nc_pcs_2020 = zeros(s);

for i=1:s(2)
    
    idx = find(abs(norm_nc_pcs_2020(:,i) - norm_era5_pcs_2020(:, i)) < 5); 
    mdl=fitlm(norm_nc_pcs_2020(idx,i), norm_era5_pcs_2020(idx, i));
    nc_adj = table2array(mdl.Coefficients(2,1))*norm_nc_pcs_2020(:,i)+table2array(mdl.Coefficients(1,1));
    
    error=norm_era5_pcs_2020(:, i)-nc_adj;
    x=1;
    for j =0:10:90
        p1=prctile(norm_era5_pcs_2020(:, i),j);
        p2=prctile(norm_era5_pcs_2020(:, i),j+10);
        ind = find((norm_era5_pcs_2020(:, i)>=p1)&(norm_era5_pcs_2020(:, i)<p2));
        pt = j;
        pte = j+10;
        pd = fitdist(error(ind),'Normal');
        se(x,i)=pd.sigma;
        me(x,i)=pd.mu;
        pb(x,i)=p1;
        
        if x == 10
            pb(11)=p2;
        end
        x=x+1;
    %p(i)=pt;
    end
    %se(:,i)=max(se(:,i));
    st(i)=nanstd(error, 0);
    in(i)=table2array(mdl.Coefficients(1,1));
    bi(i)=table2array(mdl.Coefficients(2,1));
    norm_adj_nc_pcs_2020(:,i)=nc_adj;

end

pc_error=st';
pc_slopes=bi';
pc_intercepts=in';
pc_nc_binned_error = se;
pc_nc_binned_mean = me;
pc_nc_binned_error_thresholds = pb;
pc_nc_binned_error_thresholds(1,:) = -inf;
pc_nc_binned_error_thresholds(end,:) = inf;

% change labels between u10 and v10

figure()
ax = axes(); 
hold on
h(1)=plot(norm_nc_pcs_2020(:, 1), norm_era5_pcs_2020(:, 1),'.b','DisplayName', "Before Correction");
set(gca,'fontname','times','fontsize',8)
ax=gca;
ax.FontSize = 12;
h(2)=plot(norm_adj_nc_pcs_2020(:, 1), norm_era5_pcs_2020(:, 1),'.k','DisplayName',"After Correction");
h(3)=plot([-7, 7],[-7, 7],':r', LineWidth=2, DisplayName="y = x");
grid on
%legend(,,"y=x", FontSize=12, )
%title(["Visual Crossing Bias Correction"], FontSize=14)
xlabel(["Visual Crossing", "Normalised Nowcast Principal Component 1", ], FontSize=14)
ylabel(["ERA-5","Normalised Principal Component 1"], FontSize=14)

hCopy = copyobj(h, ax); 
% replace coordinates with NaN 
% Either all XData or all YData or both should be NaN.
set(hCopy(1),'XData', NaN', 'YData', NaN)
set(hCopy(2),'XData', NaN', 'YData', NaN)
% Note, these lines can be combined: set(hCopy,'XData', NaN', 'YData', NaN)
% To avoid "Data lengths must match" warning, assuming hCopy is a handle array, 
% use arrayfun(@(h)set(h,'XData',nan(size(h.XData))),hCopy)
% Alter the graphics properties
hCopy(1).MarkerSize = 12; 
hCopy(1).LineWidth = 2;
hCopy(2).MarkerSize = 12; 
hCopy(2).LineWidth = 2; 
hCopy(3).LineWidth = 2; 
% Create legend 
legend(hCopy,Location='northwest')
xlim([-7,7])
ylim([-7,7])

%save("pc_slopes.mat", "pc_slopes")
%save("pc_intercepts.mat", "pc_intercepts")
%save("pc_error.mat","pc_error")
save("pc_nc_binned_error.mat","pc_nc_binned_error")
save("pc_nc_binned_mean.mat","pc_nc_binned_mean")
save("pc_nc_binned_error_thresholds.mat", "pc_nc_binned_error_thresholds")
