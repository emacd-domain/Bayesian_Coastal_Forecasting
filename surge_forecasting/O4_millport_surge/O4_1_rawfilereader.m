clear
clc;

filenames=[];
dinfo = dir('O4_1_millport_1980-2023\*.txt');
for K=1:length(dinfo)
  iname = string(dinfo(K).name);
  filenames=[filenames;iname];
end

for i=1:1:length(filenames)

    s=strcat('reformatting:_',string(i),'_of_', string(length(filenames)));
    disp(s)
    
    cd O4_1_millport_1980-2023

    fid=fopen(filenames(i));
    for p=1:1:12
        tline = fgetl(fid);
    end

    vararray=[];
    ii=0;

    while ischar(tline)
        ii=ii+1;
        newline = erase(tline,'T'); % remove unwanted characters
        newline = erase(tline,'R');
        TF = sum(matches(tline,'M'));
        idx = strfind(newline,')');
        newline=newline(idx+2:end);
        %formatIn='yyyy/mm/dd HH:MM:SS';
        obsdatetime=datenum(newline(1:19));%,'formatIn',formatIn);
        if TF == 0
            aline=newline(20:end);
            aline = strtrim(aline);
            a=strtok(aline);
            aa=double(string(a));
            bline=aline(length(a)+1:end);
            b=strtok(bline);
            bb=double(string(b));
        else
            aa = nan;
            bb = nan;
        end
        ALL=[obsdatetime aa bb];
        vararray=[vararray;ALL];
        tline = fgetl(fid);
        clearvars -except vararray filenames fid tline ii i
    end

    fclose(fid);

    cd ..\O4_2_reformatted

    carrot=char(filenames(i));
    redcarrot=carrot(1:end-4);
    bluecarrot=strcat(redcarrot,'.mat');
    save(bluecarrot,'vararray')

    cd ..
end