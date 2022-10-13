import cv2
import dlib
import datetime
import math
import numpy as np
import matplotlib
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import random
from imutils import face_utils
from realsense_depth import *
from scipy import array, newaxis
from sympy import Plane, Line3D
from sympy import Point3D, Plane

tiempo_actual = datetime.datetime.now()
cuadrante_actual = -1
section_time = [0,0,0,0]
activate =False
control = 0
dc = DepthCamera()
point = [0,0]
control_ojos = True
vector_CamToSubj = [0,0,0]
point_gaze = [0,0,0]
array_cara = []
valor_z = 0
angulo = 0
ax = 0
fig = plt.figure()
hog_face_detector = dlib.get_frontal_face_detector()
#Modelo entrenado
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def start_timer(current, tiempo_actual):
    
    """if(cuadrante_actual == current):
        ax += 1
    else:
        print("-----------------------------------Entra------------------------------")
        print(ax)
        print("\n\n")
        section_time[cuadrante_actual] = ax
        cuadrante_Actual = current
        ax = 1
        
    """
    print("El que miras: ", current)
    print("El que había: ", cuadrante_actual)
    if(cuadrante_actual != current):
        print("\n")
        print("-----------------------------------Entra------------------------------")
        end = datetime.datetime.now()
        section_time[current] =  end - tiempo_actual
        tiempo_actual = datetime.datetime.now()
        print(section_time[current])
        print("Cambio de cuadrante <-----")
        print("\n\n")

#Representacion del plano
def graphicRepresentationPlane(p0, p1, p2):

    #p0, p1, p2 = points
    print("Los puntos usados para el plano, son:", p0, p1, p2)
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
    vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]

    u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

    point  = np.array(p0)
    normal = np.array(u_cross_v)
    print("La normal al plano respecto a la cara es: ", normal)
    d = -point.dot(normal)

    xx, yy = np.meshgrid(range(10), range(10))

    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    # plot the surface
    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(xx, yy, z)
    #plt.show()

#Interseccion del plano de la camara con la recta de la cara
def pointIntersection(p0,v0):
    #plane Points
    a1 = Point3D (0,0,0)
    a2 = Point3D (855,915,0)
    a3 = Point3D (-805,-815,0)
    #line Points
    #p0 = Point3D (0,3,1) #point in line
    #v0 = [0, 1 ,1] #line direction as vector [normal vector]

    #create plane and line
    plane = Plane(a1,a2,a3)
    #print("The equation of the plane is: ", plane)
    line = Line3D(p0,direction_ratio=v0)


    #print(f"plane equation: {plane.equation()}")
    #print(f"line equation: {line.equation()}")

    #find intersection:

    intr = plane.intersection(line)

    intersection =np.array(intr[0],dtype=float)
    print(f"intersection: {intersection}")
    return intersection

#punto en el eje de coordenadas donde la resolucion es 640x480 y el (0,0) es el (320,240)
def getPoint(point, x, y):
    point[0] = x -320
    point[1] = y - 240
    deep = depth_frame[x,y]
    return [point[0], point[1], valueZ(x,y,deep)]

#Angulo que crea las dos rectas (No se usa)
def anguloRectas(cp, vector_CamToSubj, i):
    result_up = (cp[i]*vector_CamToSubj[i] + cp[2]*vector_CamToSubj[2])
    result_down = math.sqrt(abs(cp[i])**2 + abs(cp[2])**2) * math.sqrt(abs(vector_CamToSubj[i])**2+abs(vector_CamToSubj[2])**2)
    return result_up / result_down        

#Valor de Z con las distáncia en mm y la posicion en pixeles
def valueZ(x,y,dist):
    if(dist == 0):
        return 0
    result = (math.sqrt(abs(((x**2)+(y**2))-(dist*3.78)**2)))
    return math.trunc(result)

#(No se usa)
def tuple_values(array, index):
    l=[]
    for value in array:
       l.append(value[index])
    return l
    

while True:
    count1 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    #Activa la camra
    ret, depth_frame,frame = dc.get_frame()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Recoge el frame actual
    faces = hog_face_detector(gray)
    #Detector de cara
    detector = dlib.get_frontal_face_detector()
    i = 0
    #If para recoger el tiempo que aparece la cara pantalla
    """
    if(len(faces) > 0 and not activate):
        e = datetime.datetime.now()
        activate = True
    elif (len(faces) == 0 and activate):
            end = datetime.datetime.now()
            print("Horas minutos y segundos activo: ",end.hour - e.hour, end.minute - e.minute, end.second - e.second)
            activate = False"""
                
    for face in faces:
        
        shape = dlib_facelandmark(gray, face)
        shape = face_utils.shape_to_np(shape)
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(face)
        #Pinta el rectangulo verde
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #show the face number
        cv2.putText(frame, "Face #{}".format(i + 1), (x - 5, y),    
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #Aqui pillo los puntos
        face_landmarks = dlib_facelandmark(gray, face)
        for n in range(0, 32):
            try:
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                if(x > 480):
                    continue 
    #print("Distancia donde surge el error en X e Y --->", depth_frame[x,y], x, y, n)
                if(depth_frame[x,y] != 0):
                    profundidad = depth_frame[x,y]
                else:
                    continue
#NUEVO ---->
# n = [0:17] borde de la cara
# n = [18: 27] cejas 
# n = [28: 36] nariz
# n = [37: 48] ojos
# n = [49: 68] boca

            except:
                print(".")
            #Coge el punto medio entre los dos ojos y lo pilla como referencia
            if(valueZ((x-320),(y-240), depth_frame[x-320,y-240]) != float('0.0') ):
                if(valor_z +150 > valueZ((x-320),(y-240), depth_frame[x-320,y-240]) or valueZ((x-320),(y-240), depth_frame[x-320,y-240]) > valor_z-150):
                    #print("Entra 1")
                    valor_z = valueZ((x-320),(y-240), depth_frame[x-320,y-240])/3.78
                    #array_cara.append([math.trunc((x-320)),math.trunc((y-240)), valor_z])
                    
                    #n = posicion de puntos en el cara
                    if(count4 == 0):
                        if(n==4 or n==5 or n==6 or n==7):
                            cv2.circle(frame, (x, y), 1, (0,255,255), 1)
                            if(profundidad != 0 and profundidad != float('0.0')):
                                #print("Menton Lateral, ",profundidad, count3)
                                array_cara.append([math.trunc((x-320)),math.trunc((y-240)), valor_z])
                                #cv2.putText(frame, "Punto1 #{}".format(i + 1), (x, y),    
                                #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                count4 += 1
                    if(count5 == 0):
                        if(n==11 or n==12 or n==13 or n==14 or n == 15):
                            cv2.circle(frame, (x, y), 1, (0,255,255), 1)
                            if(profundidad != 0 and profundidad != float('0.0')):
                                #print("Menton Lateral, ",profundidad, count3)
                                #cv2.putText(frame, "Punto2 #{}".format(i + 1), (x, y),    
                                #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                array_cara.append([math.trunc((x-320)),math.trunc((y-240)), valor_z])
                                count5 += 1
                    if(count1 == 0):
                        if(n==28 or n==29 or n==30):
                            cv2.circle(frame, (x, y), 1, (0,255,255), 1)
                            if(profundidad != 0 and profundidad != float('0.0')):
                                #print("Nariz ", profundidad, count1)
                                count1 += 1
                                #cv2.putText(frame, "PuntoCentr #{}".format(i + 1), (x, y),    
                                #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                punto_central = [math.trunc((x-320)),math.trunc((y-240)), valor_z]
                                point_gaze = punto_central
                                array_cara.append([math.trunc((x-320)),math.trunc((y-240)), valor_z])
                                gaze_point = getPoint(point, x, y)
                                control = profundidad
                else:
                    #print("Entra 2")
                    array_cara.append([math.trunc((x-320)),math.trunc((y-240)), valor_z])
#NUEVO END ->
            """    
            #Coge el punto medio entre los dos ojos y lo pilla como referencia
            if(n == 27 and profundidad != float('0.0')):
                global point
                control = profundidad
                if(getPoint(point, x, y) != [0,0,0]):
                    point_gaze = getPoint(point, x, y)
                    control_ojos = False
                print("El punto medio de los ojos es: ", point_gaze)
            elif(n == 28 and profundidad != float('0.0') and point_gaze[0] != 0 and control_ojos):
                control = profundidad
                point_gaze = getPoint(point, x, y)
                print("El punto medio 2 de los ojos es:", point_gaze)
            elif(n == 29 and profundidad != float('0.0') and point_gaze[0] != 0  and control_ojos):
                control = profundidad
                point_gaze = getPoint(point, x, y)
                print("El punto medio 3 de los ojos es:", point_gaze)
            #Recoge los puntos que hacen la silueta de la cata
            if(n < 17):
                try:
                    aux = valueZ((x-320),(y-240), depth_frame[x-320,y-240])
                    print("-----------------------------")
                    print("Valor_z es: ", valor_z)
                    print("La entreada es: ", aux)
                    if(aux != float(0.0) ):
                        if(valor_z +150 > aux or aux > valor_z-150):
                            valor_z = aux #valueZ((x-320),(y-240), depth_frame[x-320,y-240])/3.78
                            array_cara.append([math.trunc((x-320)),math.trunc((y-240)), valor_z])
                        else:
                            print("Entra 2")
                            array_cara.append([math.trunc((x-320)),math.trunc((y-240)), valor_z])
                except:
                    pass
            """
            if(n < 17 or n > 27):
                    if(n < 31): #or n > 36
                        cv2.circle(frame, (x, y), 1, (0,255,255), 1)
        i += 1
       #Grafica de los puntos seleccionados y su distancia
       #Funciones del plano que forma la cara, el vector normal de esta y donde corta en el plano de la camara
        try:
            distance = valueZ((x-320),(y-240), depth_frame[point[0]-320,point[1]-240])
            cv2.putText(frame, "{}mm".format(distance), (point[0], point[1]), cv2.FONT_HERSHEY_PLAIN, 1, (245,0,0), 2)
            x_mm = math.trunc((point[0]-320))
            y_mm = math.trunc((point[1]-240))
            cv2.circle(frame, (320, 240), 4, (0,200,255)) #circulo medio
               
            print(" <" ,control/1000, ">")
            if(control < 4000 and control != 0):
                print(array_cara)
                
                Xs = tuple_values(array_cara, 0)
                Ys = tuple_values(array_cara, 1)
                Zs = tuple_values(array_cara, 2)

                ax = fig.add_subplot(111, projection='3d')

                surf = ax.plot_trisurf(Xs, Ys, Zs, cmap=cm.jet, linewidth=0)
                fig.colorbar(surf)

                ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.yaxis.set_major_locator(MaxNLocator(6))
                ax.zaxis.set_major_locator(MaxNLocator(5))

                fig.tight_layout()
                plt.show()
                
                #y = random.choice(range(round((len(array_cara)/2) - 1), round((len(array_cara)/2) +2)))
                p1 = np.array(array_cara[0])
                p2 = np.array(array_cara[1])
                p3 = np.array(array_cara[2])
                
                #print("Los puntos son: ",p1, p2, p3)
                #These two vectors are in the plane
                v1= p3 - p1
                v2= p2 - p1
                #print("Los vectores son: ", v1, v2)
                #the cross products is the vector normal to the plane
                cp = np.cross(abs(v1), abs(v2))
                #print("Mi vector normal es: ", cp)
                #vector_CamToSubj = vector_CamToSubj - cp
                #angulo_x =  anguloRectas(cp, vector_CamToSubj, 0)
                #angulo_y = anguloRectas(cp, vector_CamToSubj, 1)
                #print("El punto medio de la vista es: ", point_gaze)
                pointIntersection(point_gaze ,cp) #cambiar lo por gaze_point
#NUEVO --->                
                print("--------------------------")
                vista = pointIntersection(point_gaze ,cp)
                #Guardas la variable del cuadrante en el que miras y cada vez que cambie de cuadrante cambia de tiempo
                #Hacer una array y estimar cuanto tiempo mira en cada cuadrante cada vez que cambia de zona
                if(abs(punto_central[0] - array_cara[0][0]) < abs(punto_central[0] - array_cara[1][0])):
                    print("Esta mirando hacia la Derecha")
                    if(vista[1] > 0):
                        print("y Arriba || Primer cuadrante")
                        start_timer(0, tiempo_actual)
                        cuadrante_actual = 0
                    else:
                        print("y Abajo || Cuarto cuadrante")
                        start_timer(3, tiempo_actual)
                        cuadrante_actual = 3
                else:
                    print("Esta mirando hacia la Izquierda")
                    if(vista[1] > 0):
                        print("y Arriba || Segundo cuadrante")
                        start_timer(1, tiempo_actual)
                        cuadrante_actual = 1
                    else:
                        print("y Abajo || Tercer cuadrante")
                        start_timer(2, tiempo_actual)
                        cuadrante_actual = 2
#END NUEVO ->                
                
                a,b,c = cp
                d = np.dot(cp, p3)
                #print("The equation is {0}x + {1}y + {2}z = {3}".format(a,b,c,d))
#                print("\nLa ecuacion de la recta perpendicular al plano en el punto medio/alto es:")
#                print("x = {0} + {1}t".format(point_gaze[0], cp[0]))
#                print("y = {0} + {1}t".format(point_gaze[1], cp[1]))
#                print("z = {0} + {1}t".format(point_gaze[2], cp[2]))
            
            array_cara = []
            point_gaze = [0,0,0]
        except Exception as error:
            print("-> ", error)
            break
    array_cara = []
    cv2.imshow("Face", frame)
    key = cv2.waitKey(0)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()