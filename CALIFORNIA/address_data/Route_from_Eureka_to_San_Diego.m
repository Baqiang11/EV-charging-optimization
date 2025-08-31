apiKey = 'AIzaSyB73qYo8hPZmufDW1yUSUqmIKASUdjUVYI';

% Define the origin and destination
origin = 'Eureka, CA';
destination = 'San Diego, CA';

% Construct the directions query URL
directionsUrl = ['https://maps.googleapis.com/maps/api/directions/json?origin=', origin, '&destination=', destination, '&key=', apiKey];

% Fetch the directions data
directionsResponse = webread(directionsUrl);

% Check if the response contains route data
if isempty(directionsResponse.routes)
    error('Failed to retrieve route data. Please check your API key and query parameters.');
end

% Extract the first route leg information
legs = directionsResponse.routes(1).legs(1);
steps = legs.steps;

% Initialize data containers
latitudes = [];
longitudes = [];
distances = [];
durations = [];
speeds_kmh = [];
adjusted_energy_consumption_kwh = [];

% Vehicle parameters
vehicle_consumption_wh_per_km = 207.54; % 1 mile = 1.60934 km
time_limit_s = 3600;  % 1 hour per segment (3600 seconds)

% Accumulators for time and distance
accumulated_distance_m = 0;
accumulated_duration_s = 0;
accumulated_energy_kwh = 0;
current_lat = steps{1}.start_location.lat;
current_lng = steps{1}.start_location.lng;

% Extract latitude, longitude, distance, and duration for each step
for i = 1:length(steps)
    start_location = steps{i}.start_location;
    end_location = steps{i}.end_location;
    
    % Extract step information
    distance_m = steps{i}.distance.value;  % Meters
    duration_s = steps{i}.duration.value;  % Seconds
    speed_kmh = (distance_m / 1000) / (duration_s / 3600);
    
    % Calculate energy consumption
    energy_consumption_kwh = (distance_m / 1000) * (vehicle_consumption_wh_per_km / 1000);  % kWh
    adjusted_energy_consumption = energy_consumption_kwh * (1 + (speed_kmh - 50) * 0.01);
    
    % Accumulate time, distance, and energy for the current step
    while accumulated_duration_s + duration_s >= time_limit_s
        % Calculate the time and distance required to reach 1 hour
        remaining_time_s = time_limit_s - accumulated_duration_s;
        portion_distance_m = (remaining_time_s / duration_s) * distance_m;
        portion_energy_kwh = (remaining_time_s / duration_s) * adjusted_energy_consumption;
        
        % Save the 1-hour segment
        latitudes = [latitudes; current_lat];
        longitudes = [longitudes; current_lng];
        distances = [distances; accumulated_distance_m + portion_distance_m];
        durations = [durations; time_limit_s];
        speeds_kmh = [speeds_kmh; speed_kmh];
        adjusted_energy_consumption_kwh = [adjusted_energy_consumption_kwh; accumulated_energy_kwh + portion_energy_kwh];
        
        % Update the remaining portion
        duration_s = duration_s - remaining_time_s;
        distance_m = distance_m - portion_distance_m;
        adjusted_energy_consumption = adjusted_energy_consumption - portion_energy_kwh;
        
        % Reset accumulators and update the starting point
        accumulated_distance_m = 0;
        accumulated_duration_s = 0;
        accumulated_energy_kwh = 0;
        current_lat = end_location.lat;
        current_lng = end_location.lng;
    end
    
    % Accumulate the remaining portion
    accumulated_distance_m = accumulated_distance_m + distance_m;
    accumulated_duration_s = accumulated_duration_s + duration_s;
    accumulated_energy_kwh = accumulated_energy_kwh + adjusted_energy_consumption;
end

% Handle the last segment if it's less than 1 hour
if accumulated_duration_s > 0
    latitudes = [latitudes; current_lat];
    longitudes = [longitudes; current_lng];
    distances = [distances; accumulated_distance_m];
    durations = [durations; accumulated_duration_s];
    speeds_kmh = [speeds_kmh; accumulated_distance_m / (accumulated_duration_s / 3600)];
    adjusted_energy_consumption_kwh = [adjusted_energy_consumption_kwh; accumulated_energy_kwh];
end

% Create a table with the route data
route_data = table(latitudes, longitudes, distances, durations, speeds_kmh, adjusted_energy_consumption_kwh, ...
                   'VariableNames', {'Latitude', 'Longitude', 'Distance_m', 'Duration_s', 'Speed_kmh', 'Energy_Consumption_kWh'});

% Save the data to a file
writetable(route_data, 'route_data_eureka_to_san_diego.csv');

% Display success message
disp('Route data has been saved as route_data_eureka_to_san_diego.csv.');
