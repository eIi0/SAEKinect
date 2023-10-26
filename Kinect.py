#-*- coding:Latin-1 -*-
#objectif de la séance afficher des triangles pleins et fils de fer
from logging import root
from operator import matmul
from math import*
import sys
import math
from turtle import window_height, window_width
import numpy as np
from vispy import gloo, app 
from vispy.app import MouseEvent,KeyEvent
from vispy.util import keys
from vispy.gloo import Program, IndexBuffer
from vispy.util.transforms import perspective, translate, rotate
sys.path.insert(1, 'Z:/BUT3/SAE Kinect/SAE Kinect/Kinect/pyKinectAzure-master')
import pykinect_azure as pykinect
import matplotlib.image as mpimage


scale = 1000


#load textures
img1 = mpimage.imread('Z:\BUT3\SAE Kinect\loopython.png')
img2 = mpimage.imread('Z:\BUT3\SAE Kinect\girafe.jfif')
img3 = mpimage.imread('Z:\BUT3\SAE Kinect\Internet-Explorer-logo.jpg')
img4 = mpimage.imread('Z:\BUT3\SAE Kinect\kirikou.jpg')
img5 = mpimage.imread('Z:\BUT3\SAE Kinect\panda_roux.jpg')
img6 = mpimage.imread('Z:\BUT3\SAE Kinect\photo-de-cheval-qui-broute_6.jpg')

JointTete = {"tete":26,"nez":27,"oeil gauche":28,
             "oreille gauche":29,"oeil droit":30,"oreille droite":31}
JointBuste = {"pelvis":0,"nombril":1,"thorax":2,"cou":3}
JointBrasGauche = {"clavicule gauche":4,"epaule gauche":5,"coude gauche":6,"poignet gauche":7,
                   "main gauche":8,"doigt gauche":9,"pouce gauche":10}
JointBrasDroit = {"clavicule droit":11,"epaule droit":12,"coude droit":13,"poignet droit":14,
                   "main droit":15,"doigt droit":16,"pouce droit":17}
JointJambeGauche = {"hanche gauche":18,"genou gauche":19,"cheville gauche":20,"pied gauche":21}
JointJambeDroite = {"hanche droite":22,"genou droite":23,"cheville droite":24,"pied droite":25}
JointConnexionJambesBuste = {"hanche droite":22, "pelvis":0,"hanche gauche":18}
JointConnexionEpauleBuste = {"clavicule droit":11, "thorax":2,"clavicule gauche":4}

def normalize(v): #calcule la norme d'un vecteur numpy et le normalise
    normalized_v = v / np.sqrt(np.sum(v**2))
    return normalized_v

def distance(x1, x2, y1, y2, z1, z2):
    dist = math.sqrt(math.pow(x1 -x2, 2)+ math.pow(y1 -y2, 2) + math.pow(z1 -z2, 2))
    return dist

def lookAt(directionx,directiony,directionz):
    direction = np.array([directionx,directiony,directionz]) #conversion en numpy array
    x = normalize(direction) #on choisit un axe, ici x
    y = np.array([0.0,1.0,0.0])#conversion en numpy array
    z = np.cross(x,y) #produit vectoriel
    z = normalize(z) #on est obligé de normaliser ici
    y = np.cross(z,x) #on trouve y
    Matrix = np.transpose(np.array([x,y,z])) #en OpenGl c'est la transposée
    print(f'{Matrix}')
    return Matrix        
    


#dessin de primitives
class Triangle:
    'création d''un triangle avec couleur'
    def __init__(self,x1,y1,z1,x2,y2,z2,x3,y3,z3,program):
        self.program = program #le program que l'on va utiliser avec ces shaders
        self.program['position'] = [(x1,y1,z1),(x2,y2,z2),(x3,y3,z3)] # les vertex
        self.program['color'] = [(1,0,0,1),(0,1,0,1),(0,0,1,1)]; #la couleur de chaque vertex
        self.triangle = IndexBuffer([0,1,2]); # la topologie: ordre des vertex pour le dessin

    def draw(self):
        self.program.draw('triangles',self.triangle)

class line:
    def __init__(self,x1,y1,z1,x2,y2,z2,colR, ColG, colB ,program):
        self.program = program #le program que l'on va utiliser avec ces shaders
        self.program['position'] = [(x1,y1,z1),(x2,y2,z2)] # les vertex
        self.program['color'] = [(colR,ColG,colB,1),(colR,ColG,colB,1)]; #la couleur de chaque vertex
        self.line = IndexBuffer([0,1]); # la topologie: ordre des vertex pour le dessin
        
    def draw(self):
        self.program.draw('lines',self.line)

class TriangleWireFrame:
    'création d''un triangle en fils de fer colorés'
    def __init__(self,x1,y1,z1,x2,y2,z2,x3,y3,z3,program):
        self.program = program #le program que l'on va utiliser avec ces shaders
        self.program['position'] = [(x1,y1,z1),(x2,y2,z2),(x3,y3,z3)] # les vertex
        self.program['color'] = [(1,0,0,1),(0,1,0,1),(0,0,1,1)]; #la couleur de chaque vertex
        self.triangle = IndexBuffer([[0,1],[1,2],[2,0]]); # la topologie: ordre des vertex pour le dessin
    def draw(self):
        self.program.draw('lines',self.triangle)

class Cube_Filaire:
    def __init__(self,longueur, largeur, profondeur, colorR, ColorG, ColorB ,program):
        poslongueur = longueur/2;
        poslargeur = largeur/2;
        posprofondeur = profondeur/2;
        forme = [(-poslongueur,poslargeur,posprofondeur), (poslongueur,poslargeur,posprofondeur),(-poslongueur,-poslargeur,posprofondeur),(poslongueur,-poslargeur,posprofondeur),
        (-poslongueur,poslargeur,-posprofondeur), (poslongueur,poslargeur,-posprofondeur),(-poslongueur,-poslargeur,-posprofondeur),(poslongueur,-poslargeur,-posprofondeur)]
        self.program = program #le program que l'on va utiliser avec ces shaders
        self.program['position'] = forme
        self.program['color'] = [(colorR,ColorG,ColorB,1),(colorR,ColorG,ColorB,1),(colorR,ColorG,ColorB,1),(colorR,ColorG,ColorB,1),(colorR,ColorG,ColorB,1),(colorR,ColorG,ColorB,1),(colorR,ColorG,ColorB,1),(colorR,ColorG,ColorB,1)]; #la couleur de chaque vertex
        self.carre = IndexBuffer([[0,1],[0,2],[0,4],[5,4],[5,1], [5,7], [3,2] ,[3,7],[3,1],[6,7],[6,2], [6,4]]); # la topologie: ordre des vertex pour le dessin

    def draw(self):
        self.program.draw('lines',self.carre)

class TestColorBlueCube:
    def __init__(self,longueur, largeur, profondeur, colorR, ColorG, ColorB ,program):
        poslongueur = longueur/2;
        poslargeur = largeur/2;
        posprofondeur = profondeur/2;
        forme = [(-poslongueur,poslargeur,posprofondeur), (poslongueur,poslargeur,posprofondeur),(-poslongueur,-poslargeur,posprofondeur),(poslongueur,-poslargeur,posprofondeur),
        (-poslongueur,poslargeur,-posprofondeur), (poslongueur,poslargeur,-posprofondeur),(-poslongueur,-poslargeur,-posprofondeur),(poslongueur,-poslargeur,-posprofondeur)]
        self.program = program #le program que l'on va utiliser avec ces shaders
        self.program['position'] = forme # les vertex
        #                               1                          2                            3                     4                       5                       6                       7                           8                         9                       10                          11                      12                          
        self.program['color'] = [(colorR,ColorG,ColorB,1),(colorR,ColorG,ColorB,1),(colorR,ColorG,ColorB,1),(colorR,ColorG,ColorB,1),(colorR,ColorG,ColorB,1),(colorR,ColorG,ColorB,1),(colorR,ColorG,ColorB,1),(colorR,ColorG,ColorB,1)]; #la couleur de chaque vertex
        self.triangle = IndexBuffer([[0,1,2],[1,2,3],[4,5,6],[5,6,7],[0,2,4],[2,4,6],[1,3,5],[7,3,5],[0,1,5],[0,4,5],[2,3,7],[2,6,7]]); # la topologie: ordre des vertex pour le dessin

    def draw(self):
        self.program.draw('triangles',self.triangle)

class CubeTexture:
    def __init__(self,longueur, largeur, profondeur, program):
        poslongueur = longueur/2;
        poslargeur = largeur/2;
        posprofondeur = profondeur/2;
        forme = [(-poslongueur,poslargeur,posprofondeur),(-poslongueur,poslargeur,posprofondeur),(-poslongueur,poslargeur,posprofondeur),
                (poslongueur,poslargeur,posprofondeur),(poslongueur,poslargeur,posprofondeur),(poslongueur,poslargeur,posprofondeur),
                (-poslongueur,-poslargeur,posprofondeur),(-poslongueur,-poslargeur,posprofondeur),(-poslongueur,-poslargeur,posprofondeur),
                (poslongueur,-poslargeur,posprofondeur),(poslongueur,-poslargeur,posprofondeur),(poslongueur,-poslargeur,posprofondeur),
                (-poslongueur,poslargeur,-posprofondeur),(-poslongueur,poslargeur,-posprofondeur),(-poslongueur,poslargeur,-posprofondeur),
                (poslongueur,poslargeur,-posprofondeur), (poslongueur,poslargeur,-posprofondeur), (poslongueur,poslargeur,-posprofondeur),
                (-poslongueur,-poslargeur,-posprofondeur),(-poslongueur,-poslargeur,-posprofondeur),(-poslongueur,-poslargeur,-posprofondeur),
                (poslongueur,-poslargeur,-posprofondeur),(poslongueur,-poslargeur,-posprofondeur),(poslongueur,-poslargeur,-posprofondeur)]
        self.program = program #le program que l'on va utiliser avec ces shaders
        self.program['position'] = forme # les vertex
        
        self.program['texcoord'] = [(0,1),(1,1),(0,0),(1,1),(0,1),(1,0),(0,0),(1,0),(0,1),(1,0),(0,0),(1,1),(1,1),(0,1),(0,1),(0,1),(1,1),(1,1),(1,0),(0,0),(0,0),(0,0),(1,0),(1,0)]
        #                               1                          2                            3                     4                       5                       6                       7                           8                         9                       10                          11                      12                          
        #self.program['color'] = [(colorR,ColorG,ColorB,1),(colorR,ColorG,ColorB,1),(colorR,ColorG,ColorB,1),(colorR,ColorG,ColorB,1),(colorR,ColorG,ColorB,1),(colorR,ColorG,ColorB,1),(colorR,ColorG,ColorB,1),(colorR,ColorG,ColorB,1)]; #la couleur de chaque vertex
        
        self.Face1 = IndexBuffer([[0,3,6],[3,6,9]]);
        self.Face2 = IndexBuffer([[4,10,16],[22,10,16]]);
        self.Face3 = IndexBuffer([[12,15,18],[15,18,21]]);
        self.Face4 = IndexBuffer([[1,7,13],[7,13,19]]);
        self.Face5 = IndexBuffer([[2,5,17],[2,14,17]]);
        self.Face6 = IndexBuffer([[8,11,23],[8,20,23]]);

    def draw(self):
        self.program['texture'] = img1;
        self.program.draw('triangles',self.Face1)
        self.program['texture'] = img2;
        self.program.draw('triangles',self.Face2)
        self.program['texture'] = img3;
        self.program.draw('triangles',self.Face3)
        self.program['texture'] = img4;
        self.program.draw('triangles',self.Face4)
        self.program['texture'] = img5;
        self.program.draw('triangles',self.Face5)
        self.program['texture'] = img6;
        self.program.draw('triangles',self.Face6)


class CylindreTexture:
    def __init__(self,R, h,n, program):
        tabforme=[]
        dT = 360/n
        self.dT = dT
        self.texture = gloo.Texture2D(img4)
        self.n = n
        
        self.program = program #le program que l'on va utiliser avec ces shaders
        for T in range(0,n,int(dT)):
            x1 = R*cos((T))
            y1 = R*sin((T))
            z1 = -h/2
            x2 = R*cos((T))
            y2 = R*sin((T))
            z2 = h/2
            x3 = R*cos((T+dT))
            y3 = R*sin((T+dT))
            z3 = -h/2
            x4 = R*cos((T+dT))
            y4 = R*sin((T+dT))
            z4 = h/2
            tabforme.append([[x1,y1,z1],[x2,y2,z2],[x3,y3,z3],[x4,y4,z4]])
                        
        self.tableau = tabforme

    def draw(self):
        valVertex = self.tableau
        dT = self.dT
        n = self.n
        self.program['texture'] = img4;
        for T in range(0,n,int(dT)):
            tabdecompose = valVertex.pop(0)
            self.program['position'] = tabdecompose # les vertex
            self.Face = IndexBuffer([[0,1,2],[1,2,3]]);
            
            self.program['texcoord'] = [(T/n,0),(T/n,1),((T+1)/n,0),((T+1)/n,1)]
            
            self.program.draw('triangles',self.Face)
        
        #def (degrees):
        #    return degrees * (pi / 180)    


import math

class SphereTexture:
    def __init__(self, R, n, program):
        
        dT = 2*np.pi/n
        dP = np.pi/n
        P=0
        T=0
        c=0
        d=0
        
        self.program = program
        self.Face = IndexBuffer([[0, 1, 2], [1, 2, 3]])
        self.program['texture'] = img4
        for i in range(n):
            T=0
            for j in range(n):
                x1 = R * math.sin((P)) * math.cos((T))
                y1 = R * math.sin((P)) * math.sin((T))
                z1 = R * math.cos((P))
                x2 = R * math.sin((P + dP)) * math.cos((T))
                y2 = R * math.sin((P + dP)) * math.sin((T))
                z2 = R * math.cos((P + dP))
                x3 = R * math.sin((P)) * math.cos((T + dT))
                y3 = R * math.sin((P)) * math.sin((T + dT))
                z3 = R * math.cos((P))
                x4 = R * math.sin((P + dP)) * math.cos((T + dT))
                y4 = R * math.sin((P + dP)) * math.sin((T + dT))
                z4 = R * math.cos((P + dP))
                T=T+dT

                a=i/n
                b=(i+1)/n
                c=j/n
                d=(j+1)/n
                self.program['position'] = [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
                
                self.program['texcoord'] = [(a, c), (b, c), (a, d), (b, d)]
                
                self.program.draw('triangles', self.Face)
            P=P+dP

#def (degrees):
#    return degrees * (math.pi / 180)
    
        #self.tableau = tabforme

    #def draw(self):
    #    valVertex = self.tableau
    #    dT = self.dT
    #    n = self.n
        
    #    self.Face = IndexBuffer([[0,1,2],[1,2,3]]);
    #    #for T in range(0,int(n),int(dT)):
    #    T=0
    #    tabdecompose = valVertex.pop(0)
    #    self.program['position'] = tabdecompose # les vertex
            
            
    #    self.program['texcoord'] = [(T/n,0),(T/n,1),((T+1)/n,0),((T+1)/n,1)]
            
    #    self.program.draw('triangles',self.Face)
        
#def (degrees):
#    return degrees * (pi / 180)    




#vertex shader------------------------------
vertexColor = """
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
attribute vec3 position;
attribute vec4 color;
varying vec4 v_color;
void main()
{
    gl_Position = projection * view * model * vec4(position,1.0);
    v_color = color;
}
"""
#fragment shader---------------------------------------------------------------
fragmentColor = """
varying vec4 v_color;
void main()
{
    gl_FragColor =v_color;
}
"""

vertexTexture = """
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform sampler2D texture;
attribute vec3 position;
attribute vec2 texcoord;
attribute vec3 normal;
varying vec2 v_texcoord;
void main()
{
gl_Position = projection * view * model * vec4(position,1.0);
v_texcoord = texcoord;
}
"""
#fragment shader---------------------------------------------------------------
fragmentTexture = """
uniform sampler2D texture;
varying vec2 v_texcoord;
void main()
{
gl_FragColor = texture2D(texture, v_texcoord);
} """

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, size=(512, 512), title='World Frame',keys='interactive')
        # Build program & data
        self.numbersbody = 0
        self.program = Program(vertexColor, fragmentColor) #les shaders que l'on va utiliser
        self.programTexture = Program(vertexTexture, fragmentTexture) #les shaders que l'on va utiliser
        self.programTextureTete = Program(vertexTexture, fragmentTexture) #les shaders que l'on va utiliser
        self.programColorBody = Program(vertexColor, fragmentColor) #les shaders que l'on va utiliser
        

        #Commandes timer pour rotation
        #self.thetax = 0.0 #variable d'angle
        #self.timer = app.Timer('auto', self.on_timer) #construction d'un timer
        #self.timer.start() #lancement du timer

        #Commandes pour suivi de la souris
        self.thetax = 0.0 #variable d'angle
        self.thetay = 0.0 #variable d'angle

        # Build view, model, projection & normal
        view = translate((0, 0, -4)) #on recule la camera suivant z
        model = np.eye(4, dtype=np.float32) #matrice identitée
        self.program['model'] = model #matrice de l'objet
        self.program['view'] = view #matrice de la camera
        self.programTexture['model'] = model #matrice de l'objet
        self.programTexture['view'] = view #matrice de la camera
        self.programTextureTete['model'] = model
        self.programTextureTete['view'] = view
        self.programColorBody['model'] = model
        self.programColorBody['view'] = view

        gloo.set_state(clear_color=(0.30, 0.30, 0.35, 1.00), depth_test=True) #couleur du fond et test de profondeur
        self.activate_zoom() #generation de la matrice de projection
        self.show() #rendu
        self.timer=app.Timer('auto',self.on_timer)
        self.bodyTracker=bodyTracker
        self.device=device
        self.timer.start()






    def drawFrame(self):
    #    #t1 = Triangle(0,0,0,0,1,0,0.5,0.5,0,self.program) #construction d'un objet triangle
    #    #t1.draw() #affichage de l'objet

    #    #t2 = TriangleWireFrame(-1.,0,0,-1.,1,0,-0.5,0.5,0,self.program) #construction d'un objet triangle
    #    #t2.draw() #affichage de l'objet

        tX = line(0,0,0,1,0,0,1,0,0,self.program) #création d'un ligne en X couleur rouge 
        tX.draw()
        tY = line(0,0,0,0,1,0,0,1,0,self.program) #création d'un ligne en Y couleur vert
        tY.draw()
        tZ = line(0,0,0,0,0,1,0,0,1,self.program) #création d'un ligne en Z couleur bleue
        tZ.draw()

    #    #tX = Cube_Filaire(2,2,2,0,1,1,self.program) #création d'un ligne en X couleur rouge 
    #    #tX.draw()

    #    ##t2 = CubeTexture(1,1,1,self.programTexture) #création d'un ligne en X couleur rouge 
    #    ##t2.draw()

    #    #t3 = CylindreTexture(0.8,1,360,self.programTexture) #création d'un ligne en X couleur rouge 
    #    #t3.draw()

    #    t4 = SphereTexture(0.5,20,self.programTexture) #création d'un ligne en X couleur rouge 
    #    #t4.draw()
        
 

        
         

    def on_timer(self, event):
        # Get capture
        capture = self.device.update()
        # Get body tracker frame
        body_frame = self.bodyTracker.update()
        self.numbersbody = body_frame.get_num_bodies()

        if self.numbersbody >0:
            body = body_frame.get_body()
            self.joints3D = body.joints
            #print(f"joint[13] = {body.joints[13].position.x},{body.joints[13].position.y},{body.joints[13].position.z}")
            #print(f"joint[14] = {body.joints[14].position.x},{body.joints[14].position.y},{body.joints[14].position.z}")

            
    def drawLineCube(self):
        tX = line(0,0,0,1,0,0,1,0,0,self.program) #création d'un ligne en X couleur rouge 

    def on_draw(self, event):
        gloo.set_clear_color('grey')
        gloo.clear(color=True)

        if self.numbersbody > 0:
            for i in range(self.numbersbody):

                body_frame = self.bodyTracker.update()
                body = body_frame.get_body(i)
                self.joints3D = body.joints

                clesBrasGauche = list(JointBrasGauche.keys())
                clesBrasDroit = list(JointBrasDroit.keys())
                clesJambeGauche = list(JointJambeGauche.keys())
                clesJambeDroite = list(JointJambeDroite.keys())
                clesTorse = list(JointBuste.keys())
                clesTete = list(JointTete.keys())
                clesBusteJambes = list(JointConnexionJambesBuste.keys())
                clesBusteEpaules = list(JointConnexionEpauleBuste.keys())

                for i in range(len(clesBrasGauche) - 1):
                    cle_actuelle = clesBrasGauche[i]
                    cle_suivante = clesBrasGauche[i + 1]

                    valeur_actuelle = JointBrasGauche[cle_actuelle]
                    valeur_suivante = JointBrasGauche[cle_suivante]

                    tfilbuste = line(-(self.joints3D[valeur_actuelle].position.x)/scale,-(self.joints3D[valeur_actuelle].position.y)/scale,(self.joints3D[valeur_actuelle].position.z)/scale,-(self.joints3D[valeur_suivante].position.x)/scale,-(self.joints3D[valeur_suivante].position.y)/scale,(self.joints3D[valeur_suivante].position.z)/scale,255,255,255,self.programColorBody)
                    tfilbuste.draw()

                for i in range(len(clesBrasDroit) - 1):
                    cle_actuelle = clesBrasDroit[i]
                    cle_suivante = clesBrasDroit[i + 1]

                    valeur_actuelle = JointBrasDroit[cle_actuelle]
                    valeur_suivante = JointBrasDroit[cle_suivante]

                    tfilbuste = line(-(self.joints3D[valeur_actuelle].position.x)/scale,-(self.joints3D[valeur_actuelle].position.y)/scale,(self.joints3D[valeur_actuelle].position.z)/scale,-(self.joints3D[valeur_suivante].position.x)/scale,-(self.joints3D[valeur_suivante].position.y)/scale,(self.joints3D[valeur_suivante].position.z)/scale,255,255,255,self.programColorBody)
                    tfilbuste.draw()

                for i in range(len(clesJambeGauche) - 1):
                    cle_actuelle = clesJambeGauche[i]
                    cle_suivante = clesJambeGauche[i + 1]

                    valeur_actuelle = JointJambeGauche[cle_actuelle]
                    valeur_suivante = JointJambeGauche[cle_suivante]

                    tfilbuste = line(-(self.joints3D[valeur_actuelle].position.x)/scale,-(self.joints3D[valeur_actuelle].position.y)/scale,(self.joints3D[valeur_actuelle].position.z)/scale,-(self.joints3D[valeur_suivante].position.x)/scale,-(self.joints3D[valeur_suivante].position.y)/scale,(self.joints3D[valeur_suivante].position.z)/scale,255,255,255,self.programColorBody)
                    tfilbuste.draw()

                for i in range(len(clesJambeDroite) - 1):
                    cle_actuelle = clesJambeDroite[i]
                    cle_suivante = clesJambeDroite[i + 1]

                    valeur_actuelle = JointJambeDroite[cle_actuelle]
                    valeur_suivante = JointJambeDroite[cle_suivante]

                    tfilbuste = line(-(self.joints3D[valeur_actuelle].position.x)/scale,-(self.joints3D[valeur_actuelle].position.y)/scale,(self.joints3D[valeur_actuelle].position.z)/scale,-(self.joints3D[valeur_suivante].position.x)/scale,-(self.joints3D[valeur_suivante].position.y)/scale,(self.joints3D[valeur_suivante].position.z)/scale,255,255,255,self.programColorBody)
                    tfilbuste.draw()

                for i in range(len(clesTorse) - 1):
                    cle_actuelle = clesTorse[i]
                    cle_suivante = clesTorse[i + 1]

                    valeur_actuelle = JointBuste[cle_actuelle]
                    valeur_suivante = JointBuste[cle_suivante]

                    tfilbuste = line(-(self.joints3D[valeur_actuelle].position.x)/scale,-(self.joints3D[valeur_actuelle].position.y)/scale,(self.joints3D[valeur_actuelle].position.z)/scale,-(self.joints3D[valeur_suivante].position.x)/scale,-(self.joints3D[valeur_suivante].position.y)/scale,(self.joints3D[valeur_suivante].position.z)/scale,255,255,255,self.programColorBody)
                    tfilbuste.draw()

                for i in range(len(clesBusteEpaules) - 1):
                    cle_actuelle = clesBusteEpaules[i]
                    cle_suivante = clesBusteEpaules[i + 1]

                    valeur_actuelle = JointConnexionEpauleBuste[cle_actuelle]
                    valeur_suivante = JointConnexionEpauleBuste[cle_suivante]

                    tfilbuste = line(-(self.joints3D[valeur_actuelle].position.x)/scale,-(self.joints3D[valeur_actuelle].position.y)/scale,(self.joints3D[valeur_actuelle].position.z)/scale,-(self.joints3D[valeur_suivante].position.x)/scale,-(self.joints3D[valeur_suivante].position.y)/scale,(self.joints3D[valeur_suivante].position.z)/scale,255,255,255,self.programColorBody)
                    tfilbuste.draw()

                for i in range(len(clesBusteJambes) - 1):
                    cle_actuelle = clesBusteJambes[i]
                    cle_suivante = clesBusteJambes[i + 1]

                    valeur_actuelle = JointConnexionJambesBuste[cle_actuelle]
                    valeur_suivante = JointConnexionJambesBuste[cle_suivante]

                    tfilbuste = line(-(self.joints3D[valeur_actuelle].position.x)/scale,-(self.joints3D[valeur_actuelle].position.y)/scale,(self.joints3D[valeur_actuelle].position.z)/scale,-(self.joints3D[valeur_suivante].position.x)/scale,-(self.joints3D[valeur_suivante].position.y)/scale,(self.joints3D[valeur_suivante].position.z)/scale,255,255,255,self.programColorBody)
                    tfilbuste.draw()

                H = np.eye(4) #matrice homogene
                H[3,0] = -self.joints3D[JointTete['tete']].position.x/scale
                H[3,1] = -self.joints3D[JointTete['tete']].position.y/scale
                H[3,2] = self.joints3D[JointTete['tete']].position.z/scale
                self.programTextureTete['model'] = H

                cubetete = CubeTexture(0.3,0.3,0.3,self.programTextureTete)
                cubetete.draw()



            #point1= np.array([self.joints3D[JointBrasDroit['epaule droit']].position.x/scale, self.joints3D[JointBrasDroit['epaule droit']].position.y/scale, self.joints3D[JointBrasDroit['epaule droit']].position.z/scale])
            #point2= np.array([self.joints3D[JointBrasDroit['coude droit']].position.x/scale, self.joints3D[JointBrasDroit['coude droit']].position.y/scale, self.joints3D[JointBrasDroit['coude droit']].position.z/scale])
 
            ##print(point1)
            ##print(point2)

            #d = distance(point1[0],point1[1],point1[2],point2[0],point2[1],point2[2])/2

            #direction = np.array([self.joints3D[JointBrasDroit['epaule droit']].position.x/scale - self.joints3D[JointBrasDroit['coude droit']].position.x/scale , 
            #                      self.joints3D[JointBrasDroit['epaule droit']].position.y/scale - self.joints3D[JointBrasDroit['coude droit']].position.y/scale , 
            #                      self.joints3D[JointBrasDroit['epaule droit']].position.z/scale - self.joints3D[JointBrasDroit['coude droit']].position.z/scale])
            ###for i in len(point1):
            ###       direction[i] = point2[i]-point1[i]
            ##print(direction)

            ##directionX = self.joints3D[JointBrasDroit['coude droit']].position.x/scale - self.joints3D[JointBrasDroit['epaule droit']].position.x/scale
            ##directionY = self.joints3D[JointBrasDroit['coude droit']].position.y/scale - self.joints3D[JointBrasDroit['epaule droit']].position.y/scale
            ##directionZ = self.joints3D[JointBrasDroit['coude droit']].position.z/scale - self.joints3D[JointBrasDroit['epaule droit']].position.z/scale
            ##direction = np.array([directionX, directionY, directionZ])
            ##print(direction)
            ##print(0)

            ###direction = point2-point1
            #H = np.eye(4) #matrice homogene
            ##on estime la matrice de rotation
            #H[0:3,0:3] = lookAt(direction[0],direction[1],direction[2])
            #H = np.transpose(H) #pour opengl
            ##on ajoute les translation
            #H[3,0] = self.joints3D[JointBrasDroit['epaule droit']].position.z/scale
            #H[3,1] = -self.joints3D[JointBrasDroit['epaule droit']].position.y/scale
            #H[3,2] = self.joints3D[JointBrasDroit['epaule droit']].position.x/scale
            ##decalge de la 1/2 longueur de la aprtie du coprs
            #T = np.eye(4);
            #T[3,0]= d #decalage suivant x
            #H = np.matmul(T,H) #attention à l'ordre de multiplication

            #self.programTexture['model'] = H
            #t3 = CylindreTexture(0.05,d,360,self.programTexture) #création d'un ligne en X couleur rouge 
            #t3.draw()

            

            self.update()


            

                #print(f"Valeur actuelle: {valeur_actuelle}, Valeur suivante: {valeur_suivante}")


            #jointstart = JointBrasGauche[i]
            #for i in 6:
                #[jointstart+i]
                #[jointstart+i+1]
                


        
             
        #    tfilbuste = line((self.joints3D[0].position.x)/scale,(self.joints3D[0].position.y)/scale,(self.joints3D[0].position.z)/scale,(self.joints3D[1].position.x)/scale,(self.joints3D[1].position.y)/scale,(self.joints3D[1].position.z)/scale,255,255,255,self.program)
        #    tfilbuste.draw()
        #    tfilbuste = line((self.joints3D[1].position.x)/scale,(self.joints3D[1].position.y)/scale,(self.joints3D[1].position.z)/scale,(self.joints3D[2].position.x)/scale,(self.joints3D[2].position.y)/scale,(self.joints3D[2].position.z)/scale,255,255,255,self.program)
        #    tfilbuste.draw()
        #    tfilbuste = line((self.joints3D[2].position.x)/scale,(self.joints3D[2].position.y)/scale,(self.joints3D[2].position.z)/scale,(self.joints3D[3].position.x)/scale,(self.joints3D[3].position.y)/scale,(self.joints3D[3].position.z)/scale,255,255,255,self.program)
        #    tfilbuste.draw()
        #    tfilbuste = line((self.joints3D[3].position.x)/scale,(self.joints3D[3].position.y)/scale,(self.joints3D[3].position.z)/scale,(self.joints3D[26].position.x)/scale,(self.joints3D[26].position.y)/scale,(self.joints3D[26].position.z)/scale,255,255,255,self.program)
        #    tfilbuste.draw()
        #    tfilbuste = line((self.joints3D[11].position.x)/scale,(self.joints3D[11].position.y)/scale,(self.joints3D[11].position.z)/scale,(self.joints3D[12].position.x)/scale,(self.joints3D[12].position.y)/scale,(self.joints3D[12].position.z)/scale,255,255,255,self.program)
        #    tfilbuste.draw()
        #    tfilbuste = line((self.joints3D[12].position.x)/scale,(self.joints3D[12].position.y)/scale,(self.joints3D[12].position.z)/scale,(self.joints3D[13].position.x)/scale,(self.joints3D[13].position.y)/scale,(self.joints3D[13].position.z)/scale,255,255,255,self.program)
        #    tfilbuste.draw()
        #    tfilbuste = line((self.joints3D[13].position.x)/scale,(self.joints3D[13].position.y)/scale,(self.joints3D[13].position.z)/scale,(self.joints3D[14].position.x)/scale,(self.joints3D[14].position.y)/scale,(self.joints3D[14].position.z)/scale,255,255,255,self.program)
        #    tfilbuste.draw()
        #    tfilbuste = line((self.joints3D[15].position.x)/scale,(self.joints3D[15].position.y)/scale,(self.joints3D[15].position.z)/scale,(self.joints3D[16].position.x)/scale,(self.joints3D[16].position.y)/scale,(self.joints3D[16].position.z)/scale,255,255,255,self.program)
        #    tfilbuste.draw()
        #    tfilbuste = line((self.joints3D[14].position.x)/scale,(self.joints3D[14].position.y)/scale,(self.joints3D[14].position.z)/scale,(self.joints3D[15].position.x)/scale,(self.joints3D[15].position.y)/scale,(self.joints3D[15].position.z)/scale,255,255,255,self.program)
        #    tfilbuste.draw()
        #    tfilbuste = line((self.joints3D[15].position.x)/scale,(self.joints3D[15].position.y)/scale,(self.joints3D[15].position.z)/scale,(self.joints3D[17].position.x)/scale,(self.joints3D[17].position.y)/scale,(self.joints3D[17].position.z)/scale,255,255,255,self.program)
        #    tfilbuste.draw()
        #    tfilbuste = line((self.joints3D[18].position.x)/scale,(self.joints3D[18].position.y)/scale,(self.joints3D[18].position.z)/scale,(self.joints3D[19].position.x)/scale,(self.joints3D[19].position.y)/scale,(self.joints3D[19].position.z)/scale,255,255,255,self.program)
        #    tfilbuste.draw()
        #    tfilbuste = line((self.joints3D[19].position.x)/scale,(self.joints3D[19].position.y)/scale,(self.joints3D[19].position.z)/scale,(self.joints3D[20].position.x)/scale,(self.joints3D[20].position.y)/scale,(self.joints3D[20].position.z)/scale,255,255,255,self.program)
        #    tfilbuste.draw()
        #    tfilbuste = line((self.joints3D[20].position.x)/scale,(self.joints3D[20].position.y)/scale,(self.joints3D[20].position.z)/scale,(self.joints3D[21].position.x)/scale,(self.joints3D[21].position.y)/scale,(self.joints3D[21].position.z)/scale,255,255,255,self.program)
        #    tfilbuste.draw()
        #    tfilbuste = line((self.joints3D[JointJambeDroite["hanche droite"]].position.x)/scale,(self.joints3D[22].position.y)/scale,(self.joints3D[22].position.z)/scale,(self.joints3D[23].position.x)/scale,(self.joints3D[23].position.y)/scale,(self.joints3D[23].position.z)/scale,255,255,255,self.program)
        #    tfilbuste.draw()
        #    tfilbuste = line((self.joints3D[JointJambeDroite["genou droite"]].position.x)/scale,(self.joints3D[23].position.y)/scale,(self.joints3D[23].position.z)/scale,(self.joints3D[24].position.x)/scale,(self.joints3D[24].position.y)/scale,(self.joints3D[24].position.z)/scale,255,255,255,self.program)
        #    tfilbuste.draw()
        #    tfilbuste = line((self.joints3D[4].position.x)/scale,(self.joints3D[4].position.y)/scale,(self.joints3D[4].position.z)/scale,(self.joints3D[5].position.x)/scale,(self.joints3D[5].position.y)/scale,(self.joints3D[5].position.z)/scale,255,255,255,self.program)
        #    tfilbuste.draw()
        #    tfilbuste = line((self.joints3D[5].position.x)/scale,(self.joints3D[5].position.y)/scale,(self.joints3D[5].position.z)/scale,(self.joints3D[6].position.x)/scale,(self.joints3D[6].position.y)/scale,(self.joints3D[6].position.z)/scale,255,255,255,self.program)
        #    tfilbuste.draw()
        #    tfilbuste = line((self.joints3D[6].position.x)/scale,(self.joints3D[6].position.y)/scale,(self.joints3D[6].position.z)/scale,(self.joints3D[7].position.x)/scale,(self.joints3D[7].position.y)/scale,(self.joints3D[7].position.z)/scale,255,255,255,self.program)
        #    tfilbuste.draw()
        #    tfilbuste = line((self.joints3D[7].position.x)/scale,(self.joints3D[7].position.y)/scale,(self.joints3D[7].position.z)/scale,(self.joints3D[8].position.x)/scale,(self.joints3D[8].position.y)/scale,(self.joints3D[8].position.z)/scale,255,255,255,self.program)
        #    tfilbuste.draw()
        #    tfilbuste = line((self.joints3D[8].position.x)/scale,(self.joints3D[8].position.y)/scale,(self.joints3D[8].position.z)/scale,(self.joints3D[9].position.x)/scale,(self.joints3D[9].position.y)/scale,(self.joints3D[9].position.z)/scale,255,255,255,self.program)
        #    tfilbuste.draw()
        #    tfilbuste = line((self.joints3D[8].position.x)/scale,(self.joints3D[8].position.y)/scale,(self.joints3D[8].position.z)/scale,(self.joints3D[10].position.x)/scale,(self.joints3D[10].position.y)/scale,(self.joints3D[10].position.z)/scale,255,255,255,self.program)
        #    tfilbuste.draw()
            #self.update()


    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size) #l'écran d'affichage

    def activate_zoom(self): #pour crer la matrice de projection
        gloo.set_viewport(0, 0, *self.physical_size)
        projection = perspective(45.0, self.size[0] / float(self.size[1]),2.0, 10.0) #matrice de projection
        self.program['projection'] = projection
        self.programTexture['projection'] = projection
        self.programTextureTete['projection'] = projection
        self.programColorBody['projection'] = projection



    #def on_timer(self, event):
    #    self.thetax = self.thetax+1
    #    self.program['model'] = rotate(self.thetax, (1, 0, 0))
    #    self.update() # on remet à jour et on redessine




    #################################### - Etape 2 - 
    #def drawFrame(self):
    #    #t1 = Triangle(0,0,0,0,1,0,0.5,0.5,0,self.program) #construction d'un objet triangle
    #    #t1.draw() #affichage de l'objet

    #    #t2 = TriangleWireFrame(-1.,0,0,-1.,1,0,-0.5,0.5,0,self.program) #construction d'un objet triangle
    #    #t2.draw() #affichage de l'objet

    #    tX = line(0,0,0,1,0,0,1,0,0,self.program) #création d'un ligne en X couleur rouge 
    #    tX.draw()
    #    tY = line(0,0,0,0,1,0,0,1,0,self.program) #création d'un ligne en Y couleur vert
    #    tY.draw()
    #    tZ = line(0,0,0,0,0,1,0,0,1,self.program) #création d'un ligne en Z couleur bleue
    #    tZ.draw()

    def on_mouse_move(self,event):
        x = event.pos[0] 
        y = event.pos[1] 
        #print(self.size)               #use to know the size of the canva
        #print (event.pos)              #use to verify position reading work well
        w,h=self.size
        thetaX = (360/h)*y - 180
        thetaY = (360/w)*x - 180
        #print(w,h,thetaX,thetaY)
        Rx = rotate(thetaX, (1,0,0))
        Ry = rotate(thetaY, (0,1,0))
        R = matmul(Rx, Ry)
        #print(R)

        #self.program['model'] = R
        #self.programTexture['model'] = R
        #self.programTextureTete['model'] = R
        #self.programColorBody['model'] = R
        self.update() # on remet à jour et on redessine

       
        #self.thetax = self.mathmul(-180, 180)
        #self.program['model'] = rotate(self.thetax, (1, 0, 0))






if __name__ == "__main__":
    
    pykinect.initialize_libraries(track_body=True)
    device_config = pykinect.default_configuration
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    print("start device")
    device = pykinect.start_device(config=device_config)

    bodyTracker = pykinect.start_body_tracker()
    c = Canvas() #construction d'un objet Canvas
    app.run()