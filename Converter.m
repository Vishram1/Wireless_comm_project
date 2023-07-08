function a = arrayRespone(azimuth, N, d, lambda)
    a = zeros(N, length(azimuth));
    for i = 1:length(azimuth)
        a(:, i) = sqrt(1/N) * exp(1i * (0:N-1) * 2 * pi * d * sin(azimuth(i)) / lambda);
    end
end