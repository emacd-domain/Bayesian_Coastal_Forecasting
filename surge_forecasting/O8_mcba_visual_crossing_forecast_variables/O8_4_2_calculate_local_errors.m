% APPLY TO SEA LEVEL RPESSURE DIFFERENCE

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NOWCAST BUILD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
clc

st=[];bi=[];in=[];se=[];pr=[];p=[];

% import design matrix mean and standard deviations
load ..\O5_designmatrix\dm_mean.mat
load ..\O5_designmatrix\dm_std.mat

% import era5 principal components
load ..\O3_pca\era5_mslp_2020.mat
% import visual crossing principal components
load O8_2_visual_crossing_inputs_processed_to_era5_variables\nc_mslp_foc_2020.mat

s=size(nc_mslp_foc_2020);

% normalise principal components using mean and standard deviations
norm_nc_mslp_foc_2020 = (nc_mslp_foc_2020 - repmat(dm_mean(19), [s(1), 1]))./repmat(dm_std(19), [s(1), 1]);
norm_era5_mslp_2020 = (era5_mslp_2020.M - repmat(dm_mean(19), [s(1), 1]))./repmat(dm_std(19), [s(1), 1]);

% make adj pcs array
norm_adj_nc_mslp_2020 = zeros(s);
idx = (abs(norm_nc_mslp_foc_2020 - norm_era5_mslp_2020)<1);     
mdl=fitlm(norm_nc_mslp_foc_2020(idx), norm_era5_mslp_2020(idx));
fc_adj = table2array(mdl.Coefficients(2,1))*norm_nc_mslp_foc_2020+table2array(mdl.Coefficients(1,1));

error=norm_era5_mslp_2020-fc_adj;

x=1;
for j =0:10:90
    p1=prctile(norm_era5_mslp_2020,j);
    p2=prctile(norm_era5_mslp_2020,j+10);
    ind = find((norm_era5_mslp_2020>=p1)&(norm_era5_mslp_2020<p2));
    pt = j;
    pte = j+10;
    pd = fitdist(error(ind),'Normal');
    se(x)=pd.sigma;
    me(x)=pd.mu;    
    pb(x)=p1;
    
    if x == 10
        pb(11)=p2;
    end
    x=x+1;
%p(i)=pt;
end

%se(:)=max(se);
st=std(error, 'omitnan');
in=table2array(mdl.Coefficients(1,1));
bi=table2array(mdl.Coefficients(2,1));
norm_adj_nc_mslp_2020=fc_adj;

mslp_nc_slopes=bi';
mslp_nc_intercepts=in';
mslp_nc_error = st;
mslp_nc_binned_error = 1.5*se;
mslp_nc_binned_mean = me;
mslp_nc_binned_error_thresholds = pb;
mslp_nc_binned_error_thresholds(1) = -inf;
mslp_nc_binned_error_thresholds(end) = inf;
% change labels between u10 and v10
figure()
ax = axes(); 
hold on
h(1)=plot(norm_nc_mslp_foc_2020, norm_era5_mslp_2020,'.b','DisplayName', "Before Correction");
set(gca,'fontname','times','fontsize',8)
ax=gca;
ax.FontSize = 12;
h(2)=plot(norm_adj_nc_mslp_2020, norm_era5_mslp_2020,'.k','DisplayName',"After Correction");
h(3)=plot([-3, 6],[-3, 6],':r', LineWidth=2, DisplayName="y = x");
grid on
%legend(,,"y=x", FontSize=12, )
%title(["Visual Crossing Bias Correction"], FontSize=14)
xlabel(["Visual Crossing Nowcast", "Normalised Pressure Difference"], FontSize=14)
ylabel(["ERA-5","Normalised Pressure Difference"], FontSize=14)
xlim([-3,6])
ylim([-3,6])

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

cd O8_4_visual_crossing_pc_bias_correction

save("mslp_nc_slopes.mat", "mslp_nc_slopes")
save("mslp_nc_intercepts.mat", "mslp_nc_intercepts")
save("mslp_nc_binned_error.mat","mslp_nc_binned_error")
save("mslp_nc_binned_mean.mat","mslp_nc_binned_mean")
save("mslp_nc_error.mat","mslp_nc_error")
save("mslp_nc_binned_error_thresholds.mat", "mslp_nc_binned_error_thresholds")

% APPLY TO U10

cd ..

clear
clc

st=[];bi=[];in=[];se=[];pr=[];p=[];

% import era5 principal components
load ..\O3_pca\era5_u10_2020.mat
% import visual crossing principal components
load O8_2_visual_crossing_inputs_processed_to_era5_variables\nc_u10_foc_2020.mat

s=size(vc_u10_foc_2020);

% normalise principal components using mean and standard deviations
norm_fc_u10_foc_2020 = (vc_u10_foc_2020 - repmat(dm_mean(20), [s(1), 1]))./repmat(dm_std(20), [s(1), 1]);
norm_era5_u10_2020 = (era5_u10_2020.U - repmat(dm_mean(20), [s(1), 1]))./repmat(dm_std(20), [s(1), 1]);

% make adj pcs array
norm_adj_vc_u10_2020 = zeros(s);

mdl=fitlm(norm_fc_u10_foc_2020, norm_era5_u10_2020);
fc_adj = table2array(mdl.Coefficients(2,1))*norm_fc_u10_foc_2020+table2array(mdl.Coefficients(1,1));

error=norm_era5_u10_2020-fc_adj;
x=1;
for j =0:10:90
    p1=prctile(norm_era5_u10_2020,j);
    p2=prctile(norm_era5_u10_2020,j+10);
    ind = find((norm_era5_u10_2020>=p1)&(norm_era5_u10_2020<p2));
    pt = j;
    pte = j+10;
    pd = fitdist(error(ind),'Normal');
    se(x)=pd.sigma;
    me(x)=pd.mu;    
    pb(x)=p1;
    
    if x == 10
        pb(11)=p2;
    end
    x=x+1;
%p(i)=pt;
end

%se(:)=max(se);
st=std(error, 'omitnan');
in=table2array(mdl.Coefficients(1,1));
bi=table2array(mdl.Coefficients(2,1));
norm_adj_vc_u10_2020=fc_adj;


u10_nc_slopes=bi';
u10_nc_intercepts=in';
u10_nc_error = st;
u10_nc_binned_error = 1.5*se;
u10_nc_binned_mean = me;
u10_nc_binned_error_thresholds = pb;
u10_nc_binned_error_thresholds(1) = -inf;
u10_nc_binned_error_thresholds(end) = inf;
% change labels between u10 and v10
figure();
ax = axes(); 
hold on
h(1)=plot(norm_nc_u10_foc_2020, norm_era5_u10_2020,'.b','DisplayName', "Before Correction");
set(gca,'fontname','times','fontsize',8)
ax=gca;
ax.FontSize = 12;
h(2)=plot(norm_adj_nc_u10_2020, norm_era5_u10_2020,'.k','DisplayName',"After Correction");
h(3)=plot([-4, 4],[-4, 4],':r', LineWidth=2, DisplayName="y = x");
grid on
%legend(,,"y=x", FontSize=12, )
%title(["Visual Crossing Bias Correction"], FontSize=14)
xlabel(["Visual Crossing Nowcast", "Normalised U10"], FontSize=14)
ylabel(["ERA-5","Normalised U10"], FontSize=14)
xlim([-4,4])
ylim([-4,4])

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

cd O8_4_visual_crossing_pc_bias_correction

save("u10_nc_slopes.mat", "u10_nc_slopes")
save("u10_nc_intercepts.mat", "u10_nc_intercepts")
save("u10_nc_binned_error.mat","u10_nc_binned_error")
save("u10_nc_binned_mean.mat","u10_nc_binned_mean")
save("u10_nc_error.mat","u10_nc_error")
save("u10_nc_binned_error_thresholds.mat", "u10_nc_binned_error_thresholds")

% APPLY TO V10

cd ..

clear
clc

st=[];bi=[];in=[];se=[];pr=[];p=[];


% import era5 principal components
load ..\O3_pca\era5_v10_2020.mat
% import visual crossing principal components
load O8_2_visual_crossing_inputs_processed_to_era5_variables\nc_v10_foc_2020.mat

s=size(vc_v10_foc_2020);

% normalise principal components using mean and standard deviations
norm_nc_v10_foc_2020 = (nc_v10_foc_2020 - repmat(dm_mean(21), [s(1), 1]))./repmat(dm_std(21), [s(1), 1]);
norm_era5_v10_2020 = (era5_v10_2020.V - repmat(dm_mean(21), [s(1), 1]))./repmat(dm_std(21), [s(1), 1]);

% make adj pcs array
norm_adj_nc_v10_2020 = zeros(s);

mdl=fitlm(norm_nc_v10_foc_2020, norm_era5_v10_2020);
fc_adj = table2array(mdl.Coefficients(2,1))*norm_nc_v10_foc_2020+table2array(mdl.Coefficients(1,1));

error=norm_era5_v10_2020-fc_adj;
x=1;
for j =0:10:90
    p1=prctile(norm_era5_v10_2020,j);
    p2=prctile(norm_era5_v10_2020,j+10);
    ind = find((norm_era5_v10_2020>=p1)&(norm_era5_v10_2020<p2));
    pt = j;
    pte = j+10;
    pd = fitdist(error(ind),'Normal');
    se(x)=pd.sigma;
    me(x)=pd.mu;    
    pb(x)=p1;
    
    if x == 10
        pb(11)=p2;
    end
    x=x+1;
%p(i)=pt;
end

%se(:)=max(se);
st=std(error, 'omitnan');
in=table2array(mdl.Coefficients(1,1));
bi=table2array(mdl.Coefficients(2,1));
norm_adj_nc_v10_2020=fc_adj;

v10_slopes=bi';
v10_intercepts=in';
v10_binned_error = 1.5*se;
v10_binned_mean = me;
v10_error = st;
v10_binned_error_thresholds = pb;
v10_binned_error_thresholds(1) = -inf;
v10_binned_error_thresholds(end) = inf;
% change labels between u10 and v10

figure();
ax = axes(); 
hold on
h(1)=plot(norm_nc_v10_foc_2020, norm_era5_v10_2020,'.b','DisplayName', "Before Correction");
set(gca,'fontname','times','fontsize',8)
ax=gca;
ax.FontSize = 12;
h(2)=plot(norm_adj_nc_v10_2020, norm_era5_v10_2020,'.k','DisplayName',"After Correction");
h(3)=plot([-4, 4],[-4, 4],':r', LineWidth=2, DisplayName="y = x");
grid on
%legend(,,"y=x", FontSize=12, )
%title(["Visual Crossing Bias Correction"], FontSize=14)
xlabel(["Visual Crossing Nowcast", "Normalised V10"], FontSize=14)
ylabel(["ERA-5","Normalised V10"], FontSize=14)
xlim([-4,4])
ylim([-4,4])

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

cd O8_4_visual_crossing_pc_bias_correction

save("v10_nc_slopes.mat", "v10_nc_slopes")
save("v10_nc_intercepts.mat", "v10_nc_intercepts")
save("v10_nc_binned_error.mat","v10_nc_binned_error")
save("v10_nc_binned_mean.mat","v10_nc_binned_mean")
save("v10_nc_error.mat","v10_nc_error")
save("v10_nc_binned_error_thresholds.mat", "v10_nc_binned_error_thresholds")

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FORECAST BUILD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% APPLY TO SEA LEVEL RPESSURE DIFFERENCE

clear
clc

st=[];bi=[];in=[];se=[];pr=[];p=[];

% import visual crossing principal components
load O8_2_visual_crossing_inputs_processed_to_era5_variables\fc_mslp_foc_2020.mat

s=size(fc_mslp_foc_2020);

% normalise principal components using mean and standard deviations
norm_fc_mslp_foc_2020 = (fc_mslp_foc_2020 - repmat(dm_mean(19), [s(1), 1]))./repmat(dm_std(19), [s(1), 1]);
norm_era5_mslp_2020 = (era5_mslp_2020.M - repmat(dm_mean(19), [s(1), 1]))./repmat(dm_std(19), [s(1), 1]);

% make adj pcs array
norm_adj_fc_mslp_2020 = zeros(s);
idx = (abs(norm_fc_mslp_foc_2020 - norm_era5_mslp_2020)<1);     
mdl=fitlm(norm_fc_mslp_foc_2020(idx), norm_era5_mslp_2020(idx));
fc_adj = table2array(mdl.Coefficients(2,1))*norm_fc_mslp_foc_2020+table2array(mdl.Coefficients(1,1));

error=norm_era5_mslp_2020-fc_adj;

x=1;
for j =0:10:90
    p1=prctile(norm_era5_mslp_2020,j);
    p2=prctile(norm_era5_mslp_2020,j+10);
    ind = find((norm_era5_mslp_2020>=p1)&(norm_era5_mslp_2020<p2));
    pt = j;
    pte = j+10;
    pd = fitdist(error(ind),'Normal');
    se(x)=pd.sigma;
    me(x)=pd.mu;    
    pb(x)=p1;
    
    if x == 10
        pb(11)=p2;
    end
    x=x+1;
%p(i)=pt;
end

%se(:)=max(se);
st=std(error, 'omitnan');
in=table2array(mdl.Coefficients(1,1));
bi=table2array(mdl.Coefficients(2,1));
norm_adj_nc_mslp_2020=fc_adj;

mslp_fc_slopes=bi';
mslp_fc_intercepts=in';
mslp_fc_error = st;
mslp_fc_binned_error = 1.5*se;
mslp_fc_binned_mean = me;
mslp_fc_binned_error_thresholds = pb;
mslp_fc_binned_error_thresholds(1) = -inf;
mslp_fc_binned_error_thresholds(end) = inf;
% change labels between u10 and v10
figure()
ax = axes(); 
hold on
h(1)=plot(norm_fc_mslp_foc_2020, norm_era5_mslp_2020,'.b','DisplayName', "Before Correction");
set(gca,'fontname','times','fontsize',8)
ax=gca;
ax.FontSize = 12;
h(2)=plot(norm_adj_fc_mslp_2020, norm_era5_mslp_2020,'.k','DisplayName',"After Correction");
h(3)=plot([-3, 6],[-3, 6],':r', LineWidth=2, DisplayName="y = x");
grid on
%legend(,,"y=x", FontSize=12, )
%title(["Visual Crossing Bias Correction"], FontSize=14)
xlabel(["Visual Crossing Forecast", "Normalised Pressure Difference"], FontSize=14)
ylabel(["ERA-5","Normalised Pressure Difference"], FontSize=14)
xlim([-3,6])
ylim([-3,6])

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

cd O8_4_visual_crossing_pc_bias_correction

save("mslp_fc_slopes.mat", "mslp_fc_slopes")
save("mslp_fc_intercepts.mat", "mslp_fc_intercepts")
save("mslp_fc_binned_error.mat","mslp_fc_binned_error")
save("mslp_fc_binned_mean.mat","mslp_fc_binned_mean")
save("mslp_fc_error.mat","mslp_fc_error")
save("mslp_fc_binned_error_thresholds.mat", "mslp_fc_binned_error_thresholds")

% APPLY TO U10

cd ..

clear
clc

st=[];bi=[];in=[];se=[];pr=[];p=[];


% import visual crossing principal components
load O8_2_visual_crossing_inputs_processed_to_era5_variables\fc_u10_foc_2020.mat

s=size(vc_u10_foc_2020);

% normalise principal components using mean and standard deviations
norm_fc_u10_foc_2020 = (fc_u10_foc_2020 - repmat(dm_mean(20), [s(1), 1]))./repmat(dm_std(20), [s(1), 1]);
norm_era5_u10_2020 = (era5_u10_2020.U - repmat(dm_mean(20), [s(1), 1]))./repmat(dm_std(20), [s(1), 1]);

% make adj pcs array
norm_adj_fc_u10_2020 = zeros(s);

mdl=fitlm(norm_fc_u10_foc_2020, norm_era5_u10_2020);
fc_adj = table2array(mdl.Coefficients(2,1))*norm_fc_u10_foc_2020+table2array(mdl.Coefficients(1,1));

error=norm_era5_u10_2020-fc_adj;
x=1;
for j =0:10:90
    p1=prctile(norm_era5_u10_2020,j);
    p2=prctile(norm_era5_u10_2020,j+10);
    ind = find((norm_era5_u10_2020>=p1)&(norm_era5_u10_2020<p2));
    pt = j;
    pte = j+10;
    pd = fitdist(error(ind),'Normal');
    se(x)=pd.sigma;
    me(x)=pd.mu;    
    pb(x)=p1;
    
    if x == 10
        pb(11)=p2;
    end
    x=x+1;
%p(i)=pt;
end

%se(:)=max(se);
st=std(error, 'omitnan');
in=table2array(mdl.Coefficients(1,1));
bi=table2array(mdl.Coefficients(2,1));
norm_adj_vc_u10_2020=fc_adj;


u10_nc_slopes=bi';
u10_nc_intercepts=in';
u10_nc_error = st;
u10_nc_binned_error = 1.5*se;
u10_nc_binned_mean = me;
u10_nc_binned_error_thresholds = pb;
u10_nc_binned_error_thresholds(1) = -inf;
u10_nc_binned_error_thresholds(end) = inf;
% change labels between u10 and v10
figure();
ax = axes(); 
hold on
h(1)=plot(norm_fc_u10_foc_2020, norm_era5_u10_2020,'.b','DisplayName', "Before Correction");
set(gca,'fontname','times','fontsize',8)
ax=gca;
ax.FontSize = 12;
h(2)=plot(norm_adj_fc_u10_2020, norm_era5_u10_2020,'.k','DisplayName',"After Correction");
h(3)=plot([-4, 4],[-4, 4],':r', LineWidth=2, DisplayName="y = x");
grid on
%legend(,,"y=x", FontSize=12, )
%title(["Visual Crossing Bias Correction"], FontSize=14)
xlabel(["Visual Crossing", "Normalised U10"], FontSize=14)
ylabel(["ERA-5","Normalised U10"], FontSize=14)
xlim([-4,4])
ylim([-4,4])

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

cd O8_4_visual_crossing_pc_bias_correction

save("u10_fc_slopes.mat", "u10_fc_slopes")
save("u10_fc_intercepts.mat", "u10_fc_intercepts")
save("u10_fc_binned_error.mat","u10_fc_binned_error")
save("u10_fc_binned_mean.mat","u10_fc_binned_mean")
save("u10_fc_error.mat","u10_fc_error")
save("u10_fc_binned_error_thresholds.mat", "u10_fc_binned_error_thresholds")

% APPLY TO V10

cd ..

clear
clc

st=[];bi=[];in=[];se=[];pr=[];p=[];

% import visual crossing principal components
load O8_2_visual_crossing_inputs_processed_to_era5_variables\fc_v10_foc_2020.mat

s=size(fc_v10_foc_2020);

% normalise principal components using mean and standard deviations
norm_fc_v10_foc_2020 = (fc_v10_foc_2020 - repmat(dm_mean(21), [s(1), 1]))./repmat(dm_std(21), [s(1), 1]);
norm_era5_v10_2020 = (era5_v10_2020.V - repmat(dm_mean(21), [s(1), 1]))./repmat(dm_std(21), [s(1), 1]);

% make adj pcs array
norm_adj_fc_v10_2020 = zeros(s);

mdl=fitlm(norm_fc_v10_foc_2020, norm_era5_v10_2020);
fc_adj = table2array(mdl.Coefficients(2,1))*norm_fc_v10_foc_2020+table2array(mdl.Coefficients(1,1));

error=norm_era5_v10_2020-fc_adj;
x=1;
for j =0:10:90
    p1=prctile(norm_era5_v10_2020,j);
    p2=prctile(norm_era5_v10_2020,j+10);
    ind = find((norm_era5_v10_2020>=p1)&(norm_era5_v10_2020<p2));
    pt = j;
    pte = j+10;
    pd = fitdist(error(ind),'Normal');
    se(x)=pd.sigma;
    me(x)=pd.mu;    
    pb(x)=p1;
    
    if x == 10
        pb(11)=p2;
    end
    x=x+1;
%p(i)=pt;
end

%se(:)=max(se);
st=std(error, 'omitnan');
in=table2array(mdl.Coefficients(1,1));
bi=table2array(mdl.Coefficients(2,1));
norm_adj_nc_v10_2020=fc_adj;

v10_fc_slopes=bi';
v10_fc_intercepts=in';
v10_fc_binned_error = se;
v10_fc_binned_mean = me;
v10_fc_error = st;
v10_fc_binned_error_thresholds = pb;
v10_fc_binned_error_thresholds(1) = -inf;
v10_fc_binned_error_thresholds(end) = inf;
% change labels between u10 and v10

figure();
ax = axes(); 
hold on
h(1)=plot(norm_fc_v10_foc_2020, norm_era5_v10_2020,'.b','DisplayName', "Before Correction");
set(gca,'fontname','times','fontsize',8)
ax=gca;
ax.FontSize = 12;
h(2)=plot(norm_adj_fc_v10_2020, norm_era5_v10_2020,'.k','DisplayName',"After Correction");
h(3)=plot([-4, 4],[-4, 4],':r', LineWidth=2, DisplayName="y = x");
grid on
%legend(,,"y=x", FontSize=12, )
%title(["Visual Crossing Bias Correction"], FontSize=14)
xlabel(["Visual Crossing", "Normalised V10"], FontSize=14)
ylabel(["ERA-5","Normalised V10"], FontSize=14)
xlim([-4,4])
ylim([-4,4])

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

cd O8_4_visual_crossing_pc_bias_correction

save("v10_fc_slopes.mat", "v10_fc_slopes")
save("v10_fc_intercepts.mat", "v10_fc_intercepts")
save("v10_fc_binned_error.mat","v10_fc_binned_error")
save("v10_fc_binned_mean.mat","v10_fc_binned_mean")
save("v10_fc_error.mat","v10_fc_error")
save("v10_fc_binned_error_thresholds.mat", "v10_fc_binned_error_thresholds")