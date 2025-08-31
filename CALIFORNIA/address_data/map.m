apiKey = 'AIzaSyB73qYo8hPZmufDW1yUSUqmIKASUdjUVYI';

% Define the origin and destination
origin = 'Eureka, CA';
destination = 'San Diego, CA';

% Construct the directions query URL
directionsUrl = ['https://maps.googleapis.com/maps/api/directions/json?origin=', origin, '&destination=', destination, '&key=', apiKey];

% Fetch the directions data
directionsResponse = webread(directionsUrl);

% Extract the route legs information
legs = directionsResponse.routes.legs;

% Extract the route steps
steps = legs.steps;

% Initialize route points
latitudes = [];
longitudes = [];

% Extract the latitude and longitude for each step
for i = 1:length(steps)
    start_location = steps{i}.start_location;
    end_location = steps{i}.end_location;
    
    latitudes = [latitudes, start_location.lat, end_location.lat];
    longitudes = [longitudes, start_location.lng, end_location.lng];
end

% Get the locations of charging stations along the route
charging_latitudes = [];
charging_longitudes = [];
charging_station_names = {};

% Search for charging stations every 5 kilometers
distance_interval = 5; % 5 kilometers
for i = 1:distance_interval:length(latitudes)
    lat = latitudes(i);
    lng = longitudes(i);
    
    % Use Google Places API to search for charging stations
    placesUrl = ['https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=', num2str(lat), ',', num2str(lng), '&radius=1000&type=charging_station&key=', apiKey];
    placesResponse = webread(placesUrl);
    
    if strcmp(placesResponse.status, 'OK')
        for j = 1:length(placesResponse.results)
            result = placesResponse.results{j}; % Extract structure from cell array
            charging_latitudes = [charging_latitudes, result.geometry.location.lat];
            charging_longitudes = [charging_longitudes, result.geometry.location.lng];
            charging_station_names = [charging_station_names, result.name];
        end
    end
end

% Create a table with charging station information
charging_stations_table = table(charging_latitudes', charging_longitudes', charging_station_names', ...
    'VariableNames', {'Latitude', 'Longitude', 'StationName'});

% Save the table as a CSV file
writetable(charging_stations_table, 'charging_stations_eureka_to_san_diego.csv');

% Plot the route and charging stations
figure;
geoplot(latitudes, longitudes, '-o', 'DisplayName', 'Route');
hold on;
geoscatter(charging_latitudes, charging_longitudes, 'r', 'filled', 'DisplayName', 'Charging Stations');
title('Route from Eureka to San Diego with Charging Stations');
legend('Route', 'Charging Stations');

% Display success message
disp('Charging station information has been saved as charging_stations_eureka_to_san_diego.csv, and the plot has been generated.');
