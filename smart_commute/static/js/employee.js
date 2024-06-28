function edit_employee(emp_id)
{
    form  = document.getElementById('emp_form');
    form.action = 'edit_employee/'+emp_id;
    form.submit();
}

function show_update_form(form_id)
{
    form  = document.getElementById(form_id);
    if(form){
        $('#'+form_id).show();
    }
}


function execute_model(){

 $.ajax({
        url: '/users/execute_model_ajax/',  // Replace with the actual URL to your server endpoint
        method: 'POST',
        data: {capacity : document.getElementById('capacity').value,
         no_of_vehicles : document.getElementById('no_of_vehicles').value,
         shift : document.getElementById('shift').value,
         occupancy : document.getElementById('occupancy').value,
         },
        success: function(data) {
          // Update other fields based on the serialized product object.

            debugger;
            $("#coordinates_"+data.vehicle).html([data.coordinates]);

          // Add more fields as needed
        },
        error: function(error) {
          console.log('Error:', error);
        }

    });
    }

function fetch_routes(model1_id, id_route)
{

    $.ajax({
                    url: '/users/fetch_routes/',
                    data: {
                        'model_id': model1_id
                    },
                    success: function (data) {
                    debugger;
                        var $item = $('#'+id_route);
                        $item.empty();
                        $item.append('<option value="">Select Route</option>');
                        $.each(data, function (key, value) {
                            $item.append('<option value="' + value.vehicle_id + '">' +'Route - '+ value.vehicle_id + '</option>');
                        });
                    }
                });
}


function fetch_vehicle_wise_details(model_id, route_field_id, vehicle_id)
{
    model1_id = document.getElementById(model_id).value
    $.ajax({
                    url: '/users/fetch_model_vehicle_data/',
                    data: {
                        'model_id': model1_id,
                        'vehicle_id': vehicle_id
                    },
                    success: function (data) {
                    debugger;
                        $('#' +route_field_id+'_distance').html(data.path_data.distance )
                        $('#' +route_field_id+'_time').html(data.path_data.duration)
                        $('#' +route_field_id+'_URL').attr('href', data.path_data.URL)

                    }
                });
}

function submitForm()
{
    form = document.getElementById('route_form')
    form.submit()
}



function fetch_route_details(object)
{

    if(document.getElementById('id_model_comb1').value != ''){
        model1_id = document.getElementById('id_model_comb1').value
     }else
     {
             alert('Please select Model 1');
             object.selectedIndex= 0
             return;
     }

     if(document.getElementById('id_model_comb2').value != ''){
        model2_id = document.getElementById('id_model_comb2').value
     }else
     {
        alert('Please select Model 2');
        object.selectedIndex= 0
        return;
     }

     $.ajax({
                    url: '/users/fetch_route_details/',
                    data: {
                        'model1_id': model1_id,
                        'model2_id': model2_id,
                        'employee_id' : object.value,
                    },
                    success: function (data) {
                    debugger;
                        $('#route_model1').html(data[0].vehicle_id)
                        $('#seq_model1').html(data[0].sequence_no)
                        $('#duration_model1').html(Math.round(parseFloat(data[0].time), 2) + " Minutes")
                        $('#distance_model1').html(data[0].distance + " KM")
                        $('#route_model2').html(data[1].vehicle_id)
                        $('#seq_model2').html(data[1].sequence_no)
                        $('#duration_model2').html((Math.round(parseFloat(data[0].time), 2) +  " Minutes"))
                        $('#distance_model2').html(data[1].distance + " KM")

                    }
                });
}


 function change_model_1(){

 $.ajax({
        url: '/users/change_model/',  // Replace with the actual URL to your server endpoint
        method: 'POST',
        data: {model1 : document.getElementById('id_model_comb1').value,

         },
        success: function(data) {
          // Update other fields based on the serialized product object.

            debugger;
            $("#id_model_comb1"+data.vehicle).html([data.coordinates]);

          // Add more fields as needed
        },
        error: function(error) {
          console.log('Error:', error);
        }

    });
    }
        function initMap() {
            var mapOptions = {
                center: { lat: -34.397, lng: 150.644 },
                zoom: 8
            };
            var map = new google.maps.Map(document.getElementById('map'), mapOptions);
        }
  function loadMapScript() {
        var script = document.createElement('script');
        script.type = 'text/javascript';
        //script.src = 'https://maps.googleapis.com/maps/api/js?key=AIzaSyCk_5pj5OJufDsnZIx4YRHqlpz9fblcQ5o&libraries=places&callback=initMap';
        script.src = 'https://maps.googleapis.com/maps/api/js?key=AIzaSyDKuB9ZdvWA6BvD65w2X-P88Ejzj79_s8I&libraries=places&callback=initMap';
        document.body.appendChild(script);
    }
if(document.getElementById('map') || document.getElementById('map-static-model')){

    // Load the map script when the page is loaded
    window.onload = loadMapScript;
    }



    // Event listener for the text field
    if(document.getElementById('id_coordinates')){
        document.getElementById('id_coordinates').addEventListener('click', function() {
            $('#mapModal').modal('show');
        });

        // Initialize map when modal is shown
        $('#mapModal').on('shown.bs.modal', function () {
            google.maps.event.trigger(map, "resize");
            initMapOnClick('id_coordinates', 'id_coordinates');
        });
        }

    if(document.getElementById('id_request_type'))
    {
      document.getElementById('id_request_type').addEventListener('change', function() {
        $.ajax({
        url: '/users/change_type_onchange/',  // Replace with the actual URL to your server endpoint
        method: 'POST',
        data: {
           'type' : document.getElementById('id_request_type').value,
         },
        success: function(data) {
          // Update other fields based on the serialized product object.

            debugger;
            $("#id_old_value").val([data.old_value]);
            $("#id_old_value").prop('readonly', true);;

          // Add more fields as needed
        },
        error: function(error) {
          console.log('Error:', error);
        }

     });
     });
    }

function initMapOnClick(field_id, latlong_id) {
        var defaultLocation = { lat: 45.513743, lng: -73.583090 };
        var fieldValue = document.getElementById(field_id).value;
        if(fieldValue != '')
        {
            latlon_arr = fieldValue.split(", ");
            defaultLocation = { lat: parseFloat(latlon_arr[0]), lng: parseFloat(latlon_arr[1]) };
        }

        var map = new google.maps.Map(document.getElementById('map'), {
            center: defaultLocation,
            zoom: 12
        });

        var marker = new google.maps.Marker({
            position: defaultLocation,
            map: map,
            draggable: true
        });

        google.maps.event.addListener(marker, 'dragend', function (event) {
            var updatedLocation = {
                lat: event.latLng.lat(),
                lng: event.latLng.lng()
            };
            document.getElementById('locationInput').value = updatedLocation.lat + ', ' + updatedLocation.lng;
        });

        google.maps.event.addListener(map, 'click', function (event) {
            var clickedLocation = {
                lat: event.latLng.lat(),
                lng: event.latLng.lng()
            };

             $('#'+latlong_id).val(clickedLocation.lat + ', ' + clickedLocation.lng);
            /* getLocationName(clickedLocation.lat , clickedLocation.lng, field_id, city_field_id);*/


             marker.setPosition(clickedLocation);
             $('#mapModal').modal('hide');
        });
    }




  // Close the map modal
    function closeMapModal() {
        document.getElementById('mapModal').style.display = 'none';
    }


    function viewOnMap(points, route_type)
    {

        console.log(points);
        initMap_model(points, route_type);
    }


function initMap_model(points, route_type) {
//24.937609020683823, 67.14442239409644 //mosamiyat bus
//24.833920302842326, 67.37672438656358 // lucky
        debugger;
        $('#mapModal').modal('show');
        const center = { lat: 24.937609020683823, lng: 67.14442239409644 };
        if (route_type == 'pick'){
            or = points[0].split(", ")
            dest =  ['24.833920302842326', '67.37672438656358']
            }
        else{
            or = ['24.833920302842326', '67.37672438656358']
            dest = points[(points.length)-1].split(", ")
            }



        const origin = { lat: parseFloat(or[0]), lng: parseFloat(or[1]) } // this is the hub location
        const destination = { lat: parseFloat(dest[0]), lng: parseFloat(dest[1]) }; //route_json.destination;
        const coordinates = points; //route_json.waypoints;
        var waypoints = []
        coordinates.forEach(function(coord) {
        coord_arr = coord.split(", ");
    var location = {
        "lat": parseFloat(coord_arr[0]),
        "lng": parseFloat(coord_arr[1])
    };
    waypoints.push({"location": location, "stopover": false});
});


        // Create the map object
        const map = new google.maps.Map(document.getElementById('map'), {
          zoom: 12,
          center: center
        });

        const directionsService = new google.maps.DirectionsService();
        const directionsRenderer = new google.maps.DirectionsRenderer({suppressMarkers: true});
        directionsRenderer.setMap(map);

        // Calculate and display the route
        const calculateAndDisplayRoute = function() {
          const request = {
            origin: origin,
            destination: destination,
            waypoints: waypoints,
            optimizeWaypoints: true,
            travelMode: google.maps.TravelMode.DRIVING
          };

            debugger;
          directionsService.route(request, (result, status) => {
            if (status === google.maps.DirectionsStatus.OK) {
              directionsRenderer.setDirections(result);
            } else {
              window.alert('Directions request failed due to ' + status);
            }
          });
        }

        // Call the function to get the directions
        calculateAndDisplayRoute();

        const originMarker = new google.maps.Marker({
          position: origin,
          map: map,
          label: {
            text: "O",
            color: 'white'
          },
          title: "Starting/Ending Point"
        });

           waypoints.forEach(function(coord,index) {
                var marker = new google.maps.Marker({
                    position: coord.location,
                    map: map,
                    label :{
                     text : ''+(index+1),
                     color : 'white'
                    },
                    title : "pickup point",
                    icon: 'http://maps.google.com/mapfiles/ms/icons/green-dot.png'
                });
            });

        const destinationMarker = new google.maps.Marker({
          position: destination,
          map: map,
          label: {
            text: "D",
            color: 'white'
          },
          /*icon : "../static/img/kia_logo.svg",*/
         /* icon : "http://maps.google.com/mapfiles/ms/icons/green-dot.png",*/
          title: "Starting/Ending Point"
        });
}
