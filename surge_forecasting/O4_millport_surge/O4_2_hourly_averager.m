clear
clc;

nancount=[];

filenames=[];
dinfo = dir('O4_2_reformatted\*.mat');
for K=1:length(dinfo)
  iname = string(dinfo(K).name);
  filenames=[filenames;iname];
end

for i=1:length(filenames)
    
    s=strcat('calculating hourly averages for: ',string(i),'_of_', string(length(filenames)));
    disp(s)
       
    cd O4_2_reformatted
    load (filenames(i))

    % 13 is selected here since 15 minute resolution is only given from
    % 1993 onwards and prior to 93 hourly values are provided, hence the first 12 do not need to be averaged
    if i<13    
        hour_vararray = vararray;
    else

        hour_vararray = [];
        
        obsdatenum = vararray(:,1);
        elevation = vararray(:,2);
        residual = vararray(:,3);   
    
        for jj = 1:4:length(obsdatenum)
            obsdatenumhour = obsdatenum(jj);
            if sum(isnan(elevation(jj:jj+3)))==0 && sum(isnan(residual(jj:jj+3)))==0
                elehour=mean(elevation(jj:jj+3));
                reshour=mean(residual(jj:jj+3));
            else
                elehour=nan;
                reshour=nan;
            end
            hour_vararray=[hour_vararray;[obsdatenumhour elehour reshour]];
        end
    end
    
    a=sum(isnan(hour_vararray(:,2)));
    b=sum(isnan(hour_vararray(:,3)));
    nancount=[nancount;[a b length(hour_vararray)]];
    cd ../O4_3_hourly_averaged
    carrot=char(filenames(i));
    redcarrot=carrot(1:end-4);
    bluecarrot=strcat(redcarrot,'.mat');
    save(bluecarrot,'hour_vararray')
    cd ..
    clearvars -except filenames i nancount
end