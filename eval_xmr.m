function varargout = eval_xmr(varargin)

%   Evaluates SYSCOM files (.xmr)
%
%        [M, F] = eval_xmr()
%                 evaluates and plots the data of a single file (.xmr - MR2002CE or MR3000C, 16, 20 or 24 bit)
%                 (.smr and .vmr not yet implemented)
%                 M = file metadata
%                 F = data
%                 ==> The user will be requested to locate the data-file(s) etc. via a file selection dialog box
%
%        [M, F] = eval_xmr(chn, bg_noise, mode)
%                 M = file metadata
%                 F = data (events)
%                 chn      = 'x', 'y' or 'z' (channel used for event searching)
%                 bg_noise = x.xx (background noise in mm/s)
%                 mode     = 'single'  (evaluates all .xmr files in current folder separately)
%                            'cat'     (automatically concatenates and evaluates all MR3000C files in current folder)
%                                      (concatenation only occurs when an event is not finished at the end of a file)
%                 ==> The user will be requested to locate the data-file(s) etc. via a file selection dialog box
%              
%        [M, F] = eval_xmr(data_file, results_file, chn, bg_noise, mode)
%                 M = file metadata
%                 F = data (events)
%                 data_file = filename of data-file (pathname inclusive if not in same folder as m-file)
%                 results_file = filename of results-file (pathname inclusive if not in same folder as m-file)
%                 chn      = 'x', 'y' or 'z' (channel used for event searching)
%                 bg_noise = x.xx (background noise in mm/s)
%                 mode     = 'single'  (evaluates all .xmr files in current folder separately)
%                            'cat'     (automatically concatenates and evaluates all MR3000C files in current folder)
%                                      (concatenation only occurs when an event is not finished at the end of a file)
%
%
%   [2016-12-xx] This is not the final version, code needs to be shortened and optimized (mainly for speed)


min_duration = 4;   % Minimal duration of event in seconds
diffmax = 1.5;      % Maximal difference between files in seconds

if nargin > 5
    error('Too many input arguments!')
end

if isempty(varargin)
    [filename, pathname] = uigetfile('*', 'Please select an xmr-file...');
    f = fullfile(pathname, filename);
    if ~ischar(f) || ~exist(f, 'file')
        error('File %s does not exist!', f)
    end
    [~, ~, ext] = fileparts(filename);
    if ~strcmpi(ext, '.xmr')
        error('File %s is not an xmr-file!', f)
    end
    flag = 1;
else
    if strcmpi(varargin, 'x') + strcmpi(varargin, 'y') + strcmpi(varargin, 'z') ~= 1
        error('Check value of input argument "channel"!')
    end
    if strcmpi(varargin{1}, 'x') || strcmpi(varargin{1}, 'y') || strcmpi(varargin{1}, 'z')
        arg_in = varargin;
        % Data-file
        [filename, pathname] = uigetfile('*', 'Please select one of the data-files (.xmr)...');
        f = fullfile(pathname, filename);
        if ~ischar(f) || ~exist(f, 'file')
            error('File %s does not exist!', f)
        end
        [~, ~, ext] = fileparts(filename);
        if ~strcmpi(ext, '.xmr')
            error('File %s is not an xmr-file!', f)
        end
        % Results-file
        [filename_res, pathname_res] = uigetfile('*', 'Please select the results-file (.xls or .xlsx)...');
        f_res = fullfile(pathname_res, filename_res);
        if ~ischar(f_res) || ~exist(f_res, 'file')
            error('File %s does not exist!', f_res)
        end
        [~, ~, ext] = fileparts(filename_res);
        if (strcmp(ext, '.xls') + strcmp(ext, '.xlsx')) ~= 1
            error('File %s is not an Excel-file!', f_res)
        end    
    elseif strcmpi(varargin{3}, 'x') || strcmpi(varargin{3}, 'y') || strcmpi(varargin{3}, 'z')
        arg_in = varargin(3:end);
        % Data-file
        f = varargin{1};
        [pathname, ~, ext] = fileparts(f);
        if ~ischar(f) || ~exist(f, 'file')
            error('File %s does not exist!', f)
        end
        if ~strcmpi(ext, '.xmr')
            error('File %s is not an xmr-file!', f)
        end
        % Results-file
        f_res = varargin{2};
        [pathname_res, ~, ext] = fileparts(f_res);
        if ~ischar(f_res) || ~exist(f_res, 'file')
            error('File %s does not exist!', f_res)
        end
        if (strcmp(ext, '.xls') + strcmp(ext, '.xlsx')) ~= 1
            error('File %s is not an Excel-file!', f_res)
        end
    else
        error('Check order of input arguments!')
    end
    
    if numel(arg_in) ~= 3
        error('Too many or not enough input arguments!')
    end
    if ~isnumeric(arg_in{2})
        error('Check format of input argument "background noise"!')
    end
    if arg_in{2} < 0.001 || arg_in{2} > 2
        error('Check value of input argument "background noise"!')
    end
    
    if strcmpi(arg_in{3}, 'single')
        flag = 2;
    elseif strcmpi(arg_in{3}, 'cat')
        flag = 3;
    else
        error('Check value of last input argument!')
    end
    
    dtS_Peak = arg_in{2};
    dtS_RMS = dtS_Peak / 2.25;    % Peak / RMS ratio
    channel_num = [1, 2, 3];
    S_Channel = channel_num(strcmpi(arg_in{1}, {'x', 'y', 'z'}));
    
    Mdir = dir(pathname);
    nbentries = size(Mdir, 1);
    k = 1;
    for entry_i = 1:nbentries
        if Mdir(entry_i).isdir == false
            filename = Mdir(entry_i).name;
            if filename(1) ~= '.'
                [~, ~, ext] = fileparts(filename);
                if strcmpi(ext, '.xmr')
                    Mfiles{k} = char(filename);
                    [~, Mparts{k}, ~] = fileparts(filename);
                    k = k + 1;
                end
            end
        end
    end
    nbfiles = size(Mfiles, 2);
end

if flag == 1
    file = f;
    X = read_data_xmr(file);
    EV = X;
    FA = freq_analysis(EV);
    clear X;
    
    % Plot s(t) and single-sided amplitude spectrum of channel 1 to 3
    xlimdata = [min(EV.TimeVecNum), max(EV.TimeVecNum)];
    ymax = max([abs(min(cat(1,EV.Channel.Data))), max(cat(1,EV.Channel.Data))]);
    ylimdata = [-(ymax+0.01), ymax+0.01];
    figure
    for ii = 1:3
        subplot(3, 2, ii+(ii-1), 'align');
        plot(EV.TimeVecNum, EV.Channel(ii).Data, '-b');
        if ii == 1
            title('Vibration s(t)')
        end
        xlabel('Time [hh:mm:ss]', 'Interpreter', 'none');
        ylabel(sprintf('Channel (%d) [%s]', ii,EV.Channel(ii).Units), 'FontSize', 10, 'Interpreter', 'none');
        set(gca, 'YLim', ylimdata, 'XTick', xlimdata(1):((xlimdata(2)-xlimdata(1))/5):xlimdata(2));
        datetick('x', 13, 'keepticks')
        grid on
        grid minor
        subplot(3, 2, 2*ii, 'align');
        plot(FA.Channel(ii).Ffft, FA.Channel(ii).Yfft_amp_ri, '-b') 
        if ii == 1
            title('Single-Sided Amplitude Spectrum of s(t)')
        end
        xlabel('Frequency [Hz]', 'Interpreter', 'none');
        ylabel('|S(f)|', 'FontSize', 10, 'Interpreter', 'none');
        grid on
        grid minor    
    end
end

if flag == 2 || flag == 3
    event_number = 0;
    cont_event = 0;
    EVS = [];
    for i = 1:nbfiles
        samplediff_exc = 0;
        if flag == 2 || cont_event == 0 || i == 1
            file = fullfile(pathname, Mfiles{i});
            X = read_data_xmr(file);
            for n = 1:3
                if strcmp(X.Channel(n).Units, 'um/s') == 1
                    X.Channel(n).Data = X.Channel(n).Data / 1000;
                end
            end
            zer = num2cell(zeros(1, 6));
            [xi, xi_is, xi_isnot, delta_isnot, start_event, end_event] = zer{:}; %Set all to zero
            ndt = floor(X.nsamples / (0.2*X.SamplingRate));
            samplesdt = 0.2*X.SamplingRate;
            find_peak_dt = zeros(1, ndt);
            find_rms_dt = zeros(1, ndt);
            start_dt = 1;
        end
        if cont_event == 1
            file = fullfile(pathname, Mfiles{i});
            X_1 = X;
            X_2 = read_data_xmr(file);
            samplediff = floor((X_2.TimeVecNum(1)-X_1.TimeVecNum(end))*86400*X_1.SamplingRate) - 1;
            samplediffmax = X_1.SamplingRate*diffmax;
            if samplediff <= samplediffmax
                if samplediff < 1
                    for k = 1:3
                        X.Channel(k).Data = cat(2, X_1.Channel(k).Data, X_2.Channel(k).Data);
                    end
                else
                    for k = 1:3
                        X_gap.Channel(k).Data = X_1.Channel(k).Data(end-(samplediff-1):end);
                        X.Channel(k).Data = cat(2, cat(2, X_1.Channel(k).Data, X_gap.Channel(k).Data), X_2.Channel(k).Data);
                    end
                end
                X.nsamples = X_1.nsamples + samplediff + X_2.nsamples;
                X.dt = linspace(0, (X.nsamples/X_1.SamplingRate), X.nsamples);
                X.TimeVecNum = X_1.StartTime + X.dt/86400;
                nxdt = ndt;
                find_peak_dt_ev = find_peak_dt;
                find_rms_dt_ev = find_rms_dt;
                ndt = ndt + floor(X_2.nsamples/(0.2*X_2.SamplingRate));
                find_peak_dt = cat(2, find_peak_dt_ev,zeros(1, (ndt-nxdt)));
                find_rms_dt = cat(2, find_rms_dt_ev, zeros(1, (ndt-nxdt)));
                start_dt = nxdt + 1;
            else
                samplediff_exc = 1;
                start_dt = ndt;
            end
        end
        for dt = start_dt:ndt
            if samplediff_exc ~= 1
                peak_dt = max(abs(X.Channel(S_Channel).Data(((dt-1)*samplesdt+1):(dt*samplesdt))));
                if peak_dt >= dtS_Peak
                    find_peak_dt(dt) = 1;
                end
                rms_dt = sqrt(mean(X.Channel(S_Channel).Data(((dt-1)*samplesdt+1):(dt*samplesdt)).^2));
                if rms_dt >= dtS_RMS
                    find_rms_dt(dt) = 1;
                end
                find_event_dt = find_peak_dt(dt).*find_rms_dt(dt);
                if find_event_dt == 1
                    if start_event == 0
                        start_event = dt;
                    end
                    xi_is = xi_is + 1;
                    xi = xi + 1;
                    xi_isnot = 0;
                end
            end
            if samplediff_exc == 1 || (dt == ndt && flag == 2)
                find_event_dt = 0;
                delta_isnot = xi_isnot;
                xi_isnot = 4;
            end
            if find_event_dt == 0
                if xi_isnot <= 3 && xi >= 3
                    xi_is = xi_is + 1;
                    xi_isnot = xi_isnot + 1;
                    delta_isnot = xi_isnot;
                    if xi_isnot == 3
                        xi = 0;
                    end
                else
                    if (xi_is - delta_isnot) >= 5 * min_duration
                        end_event = dt - delta_isnot;
                        event_number = event_number + 1;
                        EV = resize_data(X, samplesdt, start_event, end_event, event_number);
                        FA = freq_analysis(EV);
                        EVS = sum_data(EV, FA, EVS);
                        start_event = 0;
                        end_event = 0;
                        xi_is = 0;
                        xi = 0;
                    else
                        start_event = 0;
                        xi_is = 0;
                        xi_isnot = 4;
                    end
                end
            end
        end
        if flag == 3 && i ~= nbfiles
            subs = str2double(Mparts{i+1}(end-4:end))-str2double(Mparts{i}(end-4:end));
            if start_event ~= 0 && end_event == 0 && subs == 1
                cont_event = 1;
            else
                cont_event = 0;
            end
        end
    end
    [EVS, MRes, MCat, SpecNum] = rewrite_data(EVS);
    clear X FA
end

varargout{1} = EV;

if flag == 1
    varargout{2} = FA;
else
    varargout{2} = EVS;
    mat_file = strcat(pathname_res, '\','eval_results_xmr.mat');
    save(mat_file, 'EVS', 'MRes', 'SpecNum');
    writetable(struct2table(MCat), f_res);
end
end


function D = read_data_xmr(file)

fid = fopen(file, 'r', 'ieee-le');

% Read fixed section of data header (256 bytes)
fseek(fid, 33, 'bof');
D.TriggerTime_IntSamp = fread(fid, 1, 'int16', 'ieee-le');
fseek(fid, 35, 'bof');
for k = 1:6
    bcd = fread(fid, 2, '2*ubit4', 'ieee-le');
    D.TriggerTime(k) = str2double(strcat(num2str(bcd(2)), num2str(bcd(1))));
end
if D.TriggerTime(6) > 10
    D.TriggerTime(6) = str2double(strcat('20', num2str(D.TriggerTime(6))));
else
    D.TriggerTime(6) = str2double(strcat('200', num2str(D.TriggerTime(6))));
end
fseek(fid, 54, 'bof');
D.SamplingRate = fread(fid, 1, 'int16', 'ieee-le');
fseek(fid, 63, 'bof');
D.ADResolution = fread(fid, 1, 'uint8', 'ieee-le');
D.Version = fread(fid, 7, '*char', 'ieee-le');
fseek(fid, 83, 'bof');
for n = 1:3
    D.Channel(n).Mantissa = fread(fid, 1, 'uint16', 'ieee-le');
    D.Channel(n).Exponent = fread(fid, 1, 'int8', 'ieee-le');
    D.Channel(n).LSB = D.Channel(n).Mantissa*10^(D.Channel(n).Exponent);    
end
fseek(fid, 92, 'bof');
for n = 1:3
    D.Channel(n).Units = fread(fid, 5, '*char', 'ieee-le');
end
for j = 1:3
    if strcmp(D.Channel(j).Units(end-3),'m') == 1
        D.Channel(j).Units = 'mm/s';
    else
        D.Channel(j).Units = 'um/s';
    end
end
fseek(fid, 143, 'bof');
D.RecordingTimePre = fread(fid, 1, 'uint8', 'ieee-le');
D.RecordingTimePost = fread(fid, 1, 'uint8', 'ieee-le');
fseek(fid, 185, 'bof');
for n = 1:3
    D.Channel(n).Offset = fread(fid, 1, 'ubit24', 'ieee-le');
end
fseek(fid, 216, 'bof');
for n = 1:3
    gain = fread(fid, 1, 'uint16', 'ieee-le');
    if gain == 0
        D.Channel(n).DSPGain = 1;
    else
        D.Channel(n).DSPGain = gain / 10000;
    end
end

% Decoding of xmr data
if D.ADResolution == 16
    for n = 1:3
        fseek(fid, 256+((n-1)*2), 'bof');
        counts = fread(fid, inf, 'bit16', 32, 'ieee-le');
        D.Channel(n).Data = D.Channel(n).LSB*counts*D.Channel(n).DSPGain;
        D.Channel(n).Length = length(D.Channel(n).Data);
        clear counts;
    end
end
if D.ADResolution == 20 || D.ADResolution == 24
    for n = 1:3
        fseek(fid, 256+((n-1)*3), 'bof');
        counts = fread(fid, inf, 'bit24', 48, 'ieee-le');
        D.Channel(n).Data = D.Channel(n).LSB*counts*D.Channel(n).DSPGain;
        D.Channel(n).Length = length(D.Channel(n).Data);
        clear counts;
    end
end
rl = unique(cat(1,D.Channel.Length));
if numel(rl) == 1
    nsamples = rl;    
else
    nsamples = min(rl);    
end
if mod(nsamples,2) == 0
    D.nsamples = nsamples;
else
    D.nsamples = nsamples - 1;
end
for n = 1:3
    D.Channel(n).Data = transpose(D.Channel(n).Data(1:D.nsamples));
    D.Channel(n).Length = D.nsamples;
end
D.dt = linspace(0, ((D.nsamples-1)/D.SamplingRate), D.nsamples);
dv = cat(2, D.TriggerTime(6:-1:4), D.TriggerTime(3:-1:1));
D.StartTime = datenum((dv.*[1,1,1,1,1,1])-[0,0,0,0,0,D.RecordingTimePre])+(D.TriggerTime_IntSamp/D.SamplingRate)/86400;
D.TimeVecNum = D.StartTime+D.dt/86400;
D.DurSec = round((D.TimeVecNum(end)-D.TimeVecNum(1))*86400+(1/D.SamplingRate), 2);

fclose(fid);
end


function N = resize_data(X, samplesdt, start_event, end_event, event_number)

% Resize data (according to event length)
N.SamplingRate = X.SamplingRate;
N.EventNumber = event_number;
N.EventStart = datestr(X.TimeVecNum((start_event-1)*samplesdt+1), 13);
N.EventEnd = datestr(X.TimeVecNum((end_event)*samplesdt), 13);
ev_start = (start_event-1)*samplesdt + 1;
ev_end = (end_event)*samplesdt;
if mod(ev_end-ev_start, 2) == 0
    ev_end = ev_end - 1;
end
for n = 1:3
   N.Channel(n).Data = X.Channel(n).Data(ev_start:ev_end);
   N.Channel(n).Units = X.Channel(n).Units;
end
N.nsamples = length(N.Channel(3).Data);
N.dt = linspace(0,((N.nsamples-1)/N.SamplingRate),N.nsamples);
N.TimeVecNum = X.TimeVecNum(((start_event-1)*samplesdt+1):((end_event)*samplesdt));
N.DurSec = round((N.TimeVecNum(end)-N.TimeVecNum(1))*86400+(1/N.SamplingRate), 2);
end


function F = freq_analysis(EV)

% Evaluation of event

% Analysis of data in time domain
for n = 1:3
    F.Channel(n).Peak = max(abs(EV.Channel(n).Data));
    F.Channel(n).RMS = sqrt(mean((EV.Channel(n).Data).^2));
end

% Analysis of data in frequenzy domain (FFT analysis)
Ns = EV.nsamples;    % 2^nextpow2(EV.nsamples);
F.nsamplesfft = Ns;
Fs = EV.SamplingRate;
dt = 1/Fs;
ffu = 1;
ffo = 80;
tau = 0.125;
KBx = 0;
HB_soll = @(f) 1./sqrt((1+((0.8*ffu)./f).^4).*(1+((0.8*f)/ffo).^4).*(1+(5.6./f).^2));
phiB_soll = @(f) atan(sqrt(2)./((f/(0.8*ffu))-((0.8*ffu)./f)))+atan(sqrt(2)./(((0.8*f)/ffo)-(ffo./(0.8*f))))+atan(5.6./f);

TSpec_raw = [3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100, 125, 160, 200, 250, 315];
FreqMax = TSpec_raw/0.89;
FreqMin = TSpec_raw/1.12;

for n = 1:3
    % FFT
    Yfft_raw = fft(EV.Channel(n).Data, Ns);
    Yfft_raw_ri = Yfft_raw(1:Ns/2+1);
    Ffft = EV.SamplingRate/2 * linspace(0, 1, Ns/2+1);
    F.Channel(n).Ffft = Ffft;
    F.Channel(n).Yfft_raw = Yfft_raw;
    F.Channel(n).Yfft_amp_ri = abs([Yfft_raw(1)/Ns, Yfft_raw(2:Ns/2)*(2/Ns), Yfft_raw((Ns/2)+1)/Ns]);
    powH = (1/(Fs*Ns)) * abs(Yfft_raw_ri).^2;
    powH(2:end-1) = 2 * powH(2:end-1);
    F.Channel(n).Yfft_pow_ri = powH;
    F.Channel(n).Yfft_pow_RSS = sqrt((1/(Ns*dt))*sum(powH));
    % KB / KBF (according to DIN 45669-1 and DIN 4150-2)
    HB = HB_soll(Ffft);
    phiB = phiB_soll(Ffft);
    H_ri = HB.*exp(phiB*1i);
    H = [H_ri(1:end),conj(flip(H_ri(2:(end-1))))];
    H_KB = Yfft_raw.*H;
    KB = abs(ifft(H_KB, Ns));
    KB_F = KB;
    for i = 1:Ns
        KBx = KBx*exp((-1*dt)/tau)+(KB(i).^2)*dt;
        KB_F(i) = sqrt((1/tau)*KBx);
    end
    F.Channel(n).KB_FTi_max = max(KB_F);
    if EV.DurSec > 30
        ntacts = floor(EV.nsamples/(EV.SamplingRate*30));
        KB_FTi_samp = [1,(1:ntacts)*(EV.SamplingRate*30), EV.nsamples];
        KB_FTi = (1:(ntacts+1));
        for i = 1:length(KB_FTi)
            KB_FTi(i) = max(KB_F(KB_FTi_samp(i):(KB_FTi_samp(i+1) - 1)));
        end
        F.Channel(n).KB_FTi_m = sqrt(mean(KB_FTi.^2));
        F.Channel(n).KB_FT_tacts = ntacts + 1;
    else
        F.Channel(n).KB_FTi_m = F.Channel(n).KB_FTi_max;
        F.Channel(n).KB_FT_tacts = 1;
    end
    nMax(n) = length(FreqMax(FreqMax < max(Ffft)));
    nMin(n) = length(FreqMin(FreqMin > min(Ffft)));
end

% T-Spectrum
TSpecMax = TSpec_raw(1:min(nMax));
TSpecMin = TSpec_raw((end-min(nMin))+1:end);
TSpec = intersect(TSpecMax, TSpecMin);
TSpecEnd = length(TSpec);

for n = 1:3
    for i = 1:TSpecEnd
        fu = FreqMin(i);
        tmpu = abs(F.Channel(n).Ffft-fu);
        [~, idxu] = min(tmpu);
        fo = FreqMax(i);
        tmpo = abs(F.Channel(n).Ffft - fo);
        [~, idxo] = min(tmpo);
        F.Channel(n).TSpecBand(1, i) = TSpec(i);
        F.Channel(n).TSpecBand(2, i) = sqrt(sum(F.Channel(n).Yfft_pow_ri(idxu:idxo))*(1/(Ns*dt)));
    end
F.Channel(n).TSpec_RSS = sqrt(sum((F.Channel(n).TSpecBand(2,1:TSpecEnd)).^2));
end
end


function S = sum_data(EV, FA, EVS)

% Summarize data of all events
S = EVS;
ev_num = EV.EventNumber;
S(ev_num).EventNumber = EV.EventNumber;
S(ev_num).Date = datestr(EV.TimeVecNum(1), 24);
S(ev_num).EventStart = EV.EventStart;
S(ev_num).EventEnd = EV.EventEnd;
S(ev_num).Duration = EV.DurSec;
S(ev_num).NSamples = EV.nsamples;
S(ev_num).SamplingRate = EV.SamplingRate;
Channel = ['x'; 'y'; 'z'];
S(ev_num).ChannelName = Channel;

S(ev_num).Peak_Data = transpose([FA.Channel.Peak]);
S(ev_num).RMS_Data = transpose([FA.Channel.RMS]);
S(ev_num).KB_FTi_max_Data = transpose([FA.Channel.KB_FTi_max]);
S(ev_num).KB_FTi_m_Data = transpose([FA.Channel.KB_FTi_m]);
S(ev_num).KB_FT_tacts = transpose([FA.Channel.KB_FT_tacts]);
S(ev_num).TSpecBand_Data = cat(1,FA.Channel.TSpecBand);
end


function [R, M, Mcat, TSpec] = rewrite_data(EVS)

% Format data
R = rmfield(EVS, 'TSpecBand_Data');
nevents = EVS(end).EventNumber;
TSpec_raw = [3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100, 125, 160, 200, 250, 315];
SpecMin = min(cat(2, EVS.TSpecBand_Data), [], 2);
SpecMax = max(cat(2, EVS.TSpecBand_Data), [], 2);
TSpec = intersect(TSpec_raw(TSpec_raw >= SpecMin(1)), TSpec_raw(TSpec_raw <= SpecMax(1)));
TSpec_cell = textscan(regexprep(num2str(TSpec), '\.', '_'), '%s');

func_tostr = @(s) textscan(sprintf('%.6f %.6f %.6f', s), '%s');
for i = 1:length(TSpec)
    field = strcat('Band_', TSpec_cell{1,1}{i});
    for n = 1:nevents
        if i == 1
            M(n).EventNumber(1:3, 1) = EVS(n).EventNumber;
            M(n).Date(1:3, 1) = {EVS(n).Date};
            M(n).Start(1:3, 1) = {EVS(n).EventStart};
            M(n).End(1:3, 1) = {EVS(n).EventEnd};
            M(n).Duration(1:3, 1) = EVS(n).Duration;
            M(n).Duration_Unit(1:3, 1) = {'[s]'};
            M(n).NSamples(1:3, 1) = EVS(n).NSamples;
            M(n).SamplR(1:3, 1) = EVS(n).SamplingRate;
            M(n).SamplR_Unit(1:3, 1) = {'[1/s]'};
            M(n).Channel = EVS(n).ChannelName;
            tempstr = func_tostr(EVS(n).Peak_Data);
            M(n).Peak = tempstr{1, 1};
            tempstr = func_tostr(EVS(n).RMS_Data);
            M(n).RMS = tempstr{1, 1};
            M(n).Vibr_Unit(1:3, 1) = {'[mm/s]'};
            tempstr = func_tostr(EVS(n).KB_FTi_max_Data);
            M(n).KBFT_max = tempstr{1, 1};
            tempstr = func_tostr(EVS(n).KB_FTi_m_Data);
            M(n).KBFT_m = tempstr{1, 1};
            M(n).KB_Unit(1:3, 1) = {'[-]'};
            M(n).KBFT_tacts = EVS(n).KB_FT_tacts;
            M(n).KB_tacts_Unit(1:3, 1) = {'[1/30s]'};
        end
        len = numel(EVS(n).TSpecBand_Data) / 6;
        if EVS(n).TSpecBand_Data(1, i) ~= TSpec(i) || i > len
            if EVS(n).TSpecBand_Data(1, i) ~= i
                EVS(n).TSpecBand_Data = [zeros(6, 1), EVS(n).TSpecBand_Data];
            end
            data = zeros(3, 1);
        else
            data = EVS(n).TSpecBand_Data(2:2:6, i);
        end
        R(n).(field) = data;
        tempstr = textscan(sprintf('%.6f %.6f %.6f', data), '%s');
        M(n).(field) = tempstr{1, 1};
        if i == length(TSpec)
            M(n).Spectr_Unit(1:3, 1) = {'[mm/s]'};
        end
    end
end
fields_M = fieldnames(M);
for k = 1:length(fields_M)
    Mcat.(fields_M{k}) = cat(1, M.(fields_M{k}));
end
end
