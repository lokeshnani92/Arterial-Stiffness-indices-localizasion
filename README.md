# Arterial-Stiffness-indices-localizasion
peaks and foot indices and inflection points and other biomarkers identification and comparision of both pulse waves with regression analysis and Bland-Altman plots
% clear all
% clc

D = load('17132520120181Z0Z0Z0.mat');
Ao = D.s_ao;
t = linspace(0, numel(Ao), numel(Ao));
[Syspks,Syslocs] = findpeaks(Ao, 'MinPeakDistance', 150, 'MinPeakHeight',-150);        % Systolic Peak
[Dcntch,Dcnlocs] = findpeaks(-Ao, 'MinPeakDistance', 40, 'MinPeakHeight',-150);        % Dicrotic Notch
[Dispks,Dislocs] = findpeaks(-Ao, 'MinPeakDistance', 150, 'MinPeakHeight',-150);       % Diastolic Peak
Dcnlocs = Dcnlocs(Ao(Dcnlocs)>0);
figure
plot(t, Ao)
hold on
plot(t(Syslocs), Syspks, '^r')
plot(t(Dcnlocs), Ao(Dcnlocs), 'vg')
plot(t(Dislocs), Ao(Dislocs), 'vk')
hold off
grid
xlim([-20 5500])
legend('Trace','Systolic Peaks', 'Dicrotic Notch', 'Diastolic Nadir', 'Location','best')


y = D.s_ao;                                                      % Data
x = linspace(0, numel(y), numel(y));                             % Create ‘x’ Vector
x = 1:numel(y);
x = x(:);

ix1 = islocalmax(y, 'MinProminence',1.1, 'MinSeparation',75);
ix2 = islocalmin(y, 'MinProminence',25,  'MinSeparation',200);
[ix3,s3,i3] = ischange(y, 'linear', 'Threshold',75);

ChgPts = [s3(ix3)  i3(ix3)  x(ix3) y(ix3)];
conds = ChgPts(:,1)>0 & ChgPts(:,1)<1;
ix3 = ChgPts(conds,3);

figure
plot(x, y)
hold on
plot(x(ix1), y(ix1), '+r')
plot(x(ix2), y(ix2), '+m')
plot(x(ix3), y(ix3), 'gp')
hold off
grid
% xlim([0 3E+3])


IDENTIFICATION OF INFLECTION POINTS OF PULSE WAVES

filepath = 'C:\Users\lokes\OneDrive\Desktop\Thesis\Thesis Data\after time interval\Aorta'
D = load('17132520120181Z0Z0Z0.mat');
y = D.s_ao;                                                      % Data
x = linspace(0, numel(y), numel(y));                                % Create ‘x’ Vector
x = 1:numel(y);
x = x(:);

ix1 = islocalmax(y, 'MinProminence',1.1, 'MinSeparation',75);
ix2 = islocalmin(y, 'MinProminence',25,  'MinSeparation',200);
[ix3,s3,i3] = ischange(y, 'linear', 'Threshold',75);

ChgPts = [s3(ix3)  i3(ix3)  x(ix3) y(ix3)];
conds = ChgPts(:,1)>0 & ChgPts(:,1)<1;
ix3 = ChgPts(conds,3);

figure
plot(x, y)
hold on
plot(x(ix1), y(ix1), '+r')
plot(x(ix2), y(ix2), '+m')
plot(x(ix3), y(ix3), 'gp')
hold off
grid
% xlim([0 3E+3])

CALCULATION OF OTHER BIOMARKERS (AUGMENTATION INDEX, PULSE PRESSURE, AUGMENTATION PRESSURE, REFLCTION INDEX, STIFFNESS INDEX, PULSE TRAANSIT TIME, MEAN ARTERIAL PRESSURE) FROM THE ABOVE SYSTOLIC PRESSURE; FOOT INDEX AND INFLECTION POINT.

% % To get the SP_ ,DP_ PP_, AI_, AP_, RI_, MAP_, PTT for Aorta and Femoral

clear all
clc

% % mark all inflection point of aorta and save them here
% % mark all inflection points of femoral and save them here
% % mark the foot index, Systolic index, of AOrta and Femoral and save them
% % save the Aorta files as [foot index(ptt_a), Inflection point(ipt_a), Systolic pressure(sp_a)]
% % save the Femoral files as [foot index(ptt_f), Inflection point(ipt_fe), Systolic pressure(sp_f)]
% % s_ao is the data or plot of Aorta
% % s_fe is the data or plot of Femoral
% % Index is the Index of both curves

load('2013161615261414171227Z11201Z0Z0Z1.mat')
dp_a=Index(ptt_a);
dp_f=Index(ptt_f);
DP_Aorta=dp_a./500;
DP_Femoral=dp_f./500;
PTT=DP_Femoral-DP_Aorta; % Pulse transit time
s_ao=s_ao'; % I have done the transpose of matrix because i need to get in 1*2335 format actually it is in 2335*1 format
s_fe=s_fe'; % I have done the transpose of matrix because i need to get in 1*2335 format actually it is in 2335*1 format
DP_AO=s_ao(ptt_a);
DP_FE=s_fe(ptt_f);
% % save PTT(Pulse Trasint time)
% % save DP_AO(Diastolic pressure Aorta)
% % save DP_FE(Diastolic pressure Femoral)

% % Ploting Systolic pressure and extracting pressure 
% % mark all systolic pressure of aorta and femoral and save them
SP_AO=s_ao(sp_a); % we get the systolic pressure of Aorta
SP_FE=s_fe(sp_f); % we get the systolic pressure of Femoral
% % save SP_AO(Systolic pressure Aorta)
% % save SP_FE(Systolic pressure Femoral)

% % Calulation of pulse pressure
PP_AO=[SP_AO-DP_AO]; % we get the pulse pressure of Aorta
PP_FE=[SP_FE-DP_FE]; % we get the pulse pressuer of Femoral
% % save PP_AO(Pulse pressure Aorta)
% % save PP_FE(Puluse pressure Femoral)

% % Calculation of Augmentation index
IPT_AO=s_ao(ipt_ao);
IPT_FE=s_fe(ipt_fe);
IP_AO=IPT_AO-DP_AO;
AP_AO=PP_AO-IP_AO; % calculation of Augmentation pressure
IP_FE=IPT_FE-DP_FE;
AP_FE=PP_FE-IP_FE; % calculation of Augmentation Pressure
AI_AO=AP_AO./PP_AO;
AI_FE=AP_FE./PP_FE;
% % save AI_AO(Augmentation index AOrta)
% % save AI_FE(Sugmentation index Femoral)
% % save AP_AO(Augmentation pressure AOrta)
% % save AP_FE(Sugmentation pressure Femoral)

% Calculation of Reflection index
RI_AO=IP_AO./PP_AO;
RI_FE=IP_FE./PP_FE;
% % save RI_AO(Refelction index Aorta)
% % save RI_FE(Refelction index Femoral)

% % Calculation of Mean Arterial Pressure
ma_ao=2*DP_AO;
map_ao=SP_AO+ma_ao;
MAP_AO=map_ao./3;
ma_fe=2*DP_FE;
map_fe=SP_FE+ma_fe;
MAP_FE=map_fe./3;
% % save MAP_AO(Mean arterial pressure Aorta)
% % save MAP_FE(MEan arterial pressure Femoral)

% % claculation of Stiffness Index
% % claculation of Stiffness Index
% tdvp_sp_ao=Index(sp_a);
% tdvp_ip_ao=Index(ipt_ao);
% tdvp_sp_fe=Index(sp_f);
% tdvp_ip_fe=Index(ipt_fe);
% Tdvp_sp_ao=tdvp_sp_ao./500;
% Tdvp_ip_ao=tdvp_ip_ao./500;
% Tdvp_sp_fe=tdvp_sp_fe./500;
% Tdvp_ip_fe=tdvp_ip_fe./500;
% INFP_PTT=Tdvp_ip_fe-Tdvp_ip_ao;% Time difference between inflection point of Femoral and Aorta
% TDVP_AO=Tdvp_sp_ao-Tdvp_ip_ao; % Time difference between systolic and Inflection point of Aorta
% TDVP_FE=Tdvp_sp_fe-Tdvp_ip_fe; % Time difference between systolic and Inflection point of Femoral
Height= 1.83 % subject height in meters
%SI_INFP=Height./INFP_PTT;
SI=Height./PTT;
% SI_AO=Height./TDVP_AO; % Stiffness Index Aorta
% SI_FE=Height./TDVP_FE; % Stiffness Index Femoral

% % save SI(Stiffness index)


% % save all files SP_ ,DP_ PP_, AI_,SI_, AP_, RI_, MAP_, PTT


REGRESSION AND BLAND-ALTMAN PLOTS COMPARISON BETWWEN TWO PULSE WAVES

clear all
clc

xdata = load('C:\Users\lokes\OneDrive\Desktop\All new from starting\Merged files\SI\SI.mat');
ydata = load('C:\Users\lokes\OneDrive\Desktop\All new from starting\Merged files\Extracted waves\SI\SI_Aorta.mat');
% Data
x = (xdata.x)';
y = (ydata.y)';
% Lineare regression line
[xyfit,GodnessFit] = fit(x,y,'poly1'); % for quadratic regression Poly2,Cubic Poly 3, 

% This will be the output of the regression analysis
% xyfit = 
% 
%      Linear model Poly1:
%      xyfit(x) = p1*x + p2
%      Coefficients (with 95% confidence bounds):
%        p1 =       2.011  (1.711, 2.311)
%        p2 =       140.9  (134.4, 147.4)

% Coefficient values of the regression model; y=p1x+p2
fprintf('The coefficient of the regression Model:')
coeff=coeffvalues(xyfit)

% Evaluation of the fitting model
fprintf(' The Godness of the regression Modell:')
R2=GodnessFit.rsquare
R=sqrt(R2)

% Plot of Regression model 
figure;
plot(xyfit,x,y,'o')
title('Regression model')

% Plot of the Residuals: the data outside the 95% range are as oultiers 
figure;
plot(xyfit,x,y,'o','residuals')
title('Residuals')
 
%%Polyfit function

[p,S] = polyfit(x,y,1); 
[y_fit,delta] = polyval(p,x,S);
figure;
plot(x,y,'bo')
hold on
plot(x,y_fit,'r-')
plot(x,y_fit+2*delta,'m--',x,y_fit-2*delta,'m--')
title('Linear Fit of Data with 95% Prediction Interval')
legend('Data','Linear Fit','95% Prediction Interval')

% Bland-Altman plot
z = (x+y);
meanAB = z./2;
diffs = x-y;
meanDiff = mean (diffs);
stdDiff = std (diffs);

meanp2D = meanDiff + 1.96 * stdDiff;
meanm2D = meanDiff- 1.96 * stdDiff;
n = length (diffs);
minD = min (meanAB) -0.1;
maxD = max (meanAB) +0.1;

figure;
plot (meanAB, diffs, '. k')
hold on;
plot ([minD; maxD], ones (1,2) * meanp2D, '- k');
text (minD + 0.01, meanp2D + 0.01, 'Mean + 1.96 * SD');
hold on;
plot ([minD; maxD], ones (1,2) * meanm2D, '- k');
text (minD + 0.01, meanm2D + 0.01, 'Mean - 1.96 * SD');
hold on;
plot ([minD; maxD], ones (1,2) * meanDiff, '- k');
text (minD + 0.01, meanDiff + 0.01, 'Mean = ');
text (minD + 0.5, meanDiff + 0.5, 'SD = ');
title('Bland-Altman plot');
xlim ([minD maxD]);
xlabel ('Average of Aorta and Femoral');
ylabel ('Aorta - Femoral');


FOR MERGING THE DIFFERENT FILES 1*N matrix TO 1 FILE
filepath = 'C:\Users\lokes\OneDrive\Desktop\Files in numbers\AI\AI_Aorta'
N = 8;
C = cell(1,N); % preallocate
for k = 1:N
    F = sprintf('AI_%u.mat',k);
    S = load(F);
    C{k} = S.AI_AO; % using the same field/variable name = very good data design!
end
x = [C{:}]; % concatenate
save('AI.mat','x')
