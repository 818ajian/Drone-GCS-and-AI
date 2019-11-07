from dronekit import connect

import dronekit_sitl
def globals_vehicle_init():
    """
    USE dronkit-sitl :
    dronekit-sitl copter --home=51.503667218218546,-0.10445594787597656,584,353 (lat,lon,alt,yaw)
    """
    global vehicle
    print('---------------    SITL   ---------------------')          
    #sitl = dronekit_sitl.start_default(lon=120.36621093749999,lat=23.483400654325642)
    sitl = dronekit_sitl.start_default(lat = 25.148266188681344, lon = 121.78979873657227)
    #connection_string = 'tcp:127.0.0.1:5760'
    connection_string = sitl.connection_string()        
    print('Connecting to vehicle on: %s' % connection_string)
    vehicle = connect(connection_string, wait_ready=True, baud=115200)  
def object_identify_init():
    global trash_num 
    global cap_num 
    trash_num = 0
    cap_num =0