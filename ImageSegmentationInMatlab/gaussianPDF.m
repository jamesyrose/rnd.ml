function vals = gaussianPDF(x, mu, std)
    values = zeros(length(x), 1);
    for i=1:length(x)
        values(i) = 1/ (std * sqrt(2 * pi)) * exp(-.5 * (x(i) - mu).^2 / std.^2);
    end
    vals = values;
end