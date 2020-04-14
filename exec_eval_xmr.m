function varargout = exec_eval_xmr(varargin)

%   Evaluates SYSCOM files (.xmr) located in different folders
%
%        F = exec_eval_xmr(X1, X2, ..., Xn)
%            F  = data
%            Xi = input arguments as a cell array {chn, bg_noise, mode},
%                 see eval_xmr.m for further information


for i = 1:nargin
    arg_in = varargin{i};
    if length(arg_in) ~= 3
        error('Cell array %s - Too many or not enough input arguments!',i)
    end
    if strcmpi(arg_in,'x') + strcmpi(arg_in,'y') + strcmpi(arg_in,'z') ~= 1
        error('Cell array %s - Check value of input argument "channel" or order of input arguments!',i)
    end
    if ~isnumeric(arg_in{2})
        error('Cell array %s - Check format of input argument "background noise"!',i)
    end
    if arg_in{2} < 0.001 || arg_in{2} > 2
        error('Cell array %s - Check value of input argument "background noise"!',i)
    end
    if strcmpi(arg_in{3},'single') + strcmpi(arg_in{3},'cat') ~= 1
        error('Cell array %s - Check value of last input argument!',i)
    end
    % Data-file
    [filename,pathname]=uigetfile('*','Please select one of the data-files (.xmr)...');
    f = fullfile(pathname,filename);
    if ~ischar(f) || ~exist(f,'file')
        error('File %s does not exist!',f)
    end
    [~,~,ext] = fileparts(filename);
    if ~strcmpi(ext,'.xmr')
        error('File %s is not an xmr-file!',f)
    end
    % Results-file
    [filename_res,pathname_res]=uigetfile('*','Please select the results-file (.xls or .xlsx)...');
    f_res = fullfile(pathname_res,filename_res);
    if ~ischar(f_res) || ~exist(f_res,'file')
        error('File %s does not exist!',f_res)
    end
    [~,~,ext] = fileparts(filename_res);
    if (strcmp(ext,'.xls') + strcmp(ext,'.xlsx')) ~= 1
        error('File %s is not an Excel-file!',f_res)
    end
    input_n{i} = {f,f_res,arg_in{1:end}};
end

for n = 1:nargin
    [~,F(n).Results] = eval_xmr(input_n{n}{1:end});
end

varargout{1} = F;

end
