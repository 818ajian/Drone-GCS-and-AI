var mymap = L.map('mapid').setView([25.146042152061465 ,121.79050683975221], 16);
L.tileLayer('https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}', {
    attribution: 'Map data &copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors, <a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Imagery © <a href="https://www.mapbox.com/">Mapbox</a>',
    maxZoom: 18,
    id: 'mapbox.streets',
    accessToken: 'pk.eyJ1Ijoic3RldmVsaWFvMTY4OCIsImEiOiJjazE3OGYyeDkxY3hsM250amdvdjBqMGFvIn0.Lv9UAvyh3NYLGGz1WGy3Ag' //ENTER YOUR ACCESS TOKEN HERE
}).addTo(mymap);

mapMarkers1 = [];
mapMarkers2 = [];
mapMarkers3 = [];
path = [];

var source = new EventSource('/topic/gcs'); //ENTER YOUR TOPICNAME HERE

source.addEventListener('message', function(e){
  console.log('Message');
  obj = JSON.parse(e.data);
  console.log(obj);

  if(obj.channel == '00001') {   
    //Mark the 'TRASH','CAP' if it is identified by yolov3 
    trash_marker = L.circleMarker([obj.latitude, obj.longitude], {
      color: 'green',
      fillColor: '#f03',
      fillOpacity: 0.5,
      radius: 3 }).addTo(mymap); 

    trash_marker.bindTooltip("lon:"+obj.longitude
      +"<br>lat:"+obj.latitude+"<br>trash:"+obj.trash_num+"<br>cap:"+obj.cap_num+"<br>time:"+obj.timestamp).openTooltip();
    mapMarkers1.push(trash_marker);    
    //trash_marker.bindTooltip("lon:"+obj.longitude).openTooltip();
  }

  if(obj.channel == '00002') {
    //mark the current copter position
    for (var i = 0; i < mapMarkers2.length; i++) {
      mymap.removeLayer(mapMarkers2[i]);
    }
    marker = L.marker([obj.latitude, obj.longitude]).addTo(mymap);
    //marker.bindPopup("lon:"+obj.longitude+"<br>lat:"+obj.latitude).openPopup();
    
    mapMarkers2.push(marker);
  }
  if(obj.channel == '00003') {
    //mark the path on map
    /*for (var i = 0; i < hmapMarkers3.lengt; i+  +) {
      mymap.removeLayer(mapMarkers3[i]);
    }*/
    circle = L.circleMarker([obj.latitude, obj.longitude], {
      color: 'red',
      fillColor: '#f03',
      fillOpacity: 0.5,
      radius: 5 }).addTo(mymap);   
    path.push([obj.latitude, obj.longitude]);    
    /*
    if(obj.path_is_ok==1){//plot path
      polyline =  L.polyline(path, {color: 'pink',weight:5,opacity:0.3,}).addTo(mymap);
    }*/
    circle.bindTooltip("WP"+ obj.waypoint).openTooltip();
  }
  if(obj.channel == '00004') {
    //mark the HOME   
    home = L.circleMarker([obj.latitude, obj.longitude], {
      color: 'green',
      fillColor: '#f03',
      fillOpacity: 0.4,
      radius: 10 }).addTo(mymap);   
       
    home.bindTooltip("HOME").openTooltip();
  }
}, false);
