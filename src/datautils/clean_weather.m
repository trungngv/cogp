clear all; clc

csvs = {'newbra.csv','newcam.csv','newchi.csv','newsot.csv'};
mats = {'bra','cam','chi','sot'};
day = 1:15;
hour = 0:23;
minute = 0:5:55;
for i=1:numel(csvs)
  xy = load(csvs{i});
  data = [];
  for d = day
    for h = hour
      for m = minute
        thetime = [d,h,m];
        rows = findRows(xy(:,1:3),thetime);
        assert(sum(rows) <= 1)
        if sum(rows) == 1
          data = [data; xy(rows,4:end)];
        else
          data = [data; -1*ones(1,4)];
        end
      end
    end
  end
  disp('missing data: ')
  disp(sum(data(:,1)==-1));
  x = (1:size(data,1))'/(24*12);
  y = data;
  save(['data/weather/' mats{i} '.mat'],'x','y')
end

