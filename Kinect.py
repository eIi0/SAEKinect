#-*- coding:Latin-1 -*-
#objectif de la séance afficher des triangles pleins et fils de fer
from operator import matmul
import sys
from turtle import window_height, window_width
import numpy as np
from vispy import gloo, app
from vispy.app import MouseEvent,KeyEvent
from vispy.util import keys
from vispy.gloo import Program, IndexBuffer
from vispy.util.transforms import perspective, translate, rotate


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

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, size=(512, 512), title='World Frame',keys='interactive')
        # Build program & data
        self.program = Program(vertexColor, fragmentColor) #les shaders que l'on va utiliser
        

        #Commandes timer pour rotation
        #self.thetax = 0.0 #variable d'angle
        #self.timer = app.Timer('auto', self.on_timer) #construction d'un timer
        #self.timer.start() #lancement du timer

        #Commandes pour suivi de la souris
        self.thetax = 0.0 #variable d'angle
        self

        # Build view, model, projection & normal
        view = translate((0, 0, -4)) #on recule la camera suivant z
        model = np.eye(4, dtype=np.float32) #matrice identitée
        self.program['model'] = model #matrice de l'objet
        self.program['view'] = view #matrice de la camera

        gloo.set_state(clear_color=(0.30, 0.30, 0.35, 1.00), depth_test=True) #couleur du fond et test de profondeur
        self.activate_zoom() #generation de la matrice de projection
        self.show() #rendu

    def drawFrame(self):
        #t1 = Triangle(0,0,0,0,1,0,0.5,0.5,0,self.program) #construction d'un objet triangle
        #t1.draw() #affichage de l'objet

        #t2 = TriangleWireFrame(-1.,0,0,-1.,1,0,-0.5,0.5,0,self.program) #construction d'un objet triangle
        #t2.draw() #affichage de l'objet

        tX = line(0,0,0,1,0,0,1,0,0,self.program) #création d'un ligne en X couleur rouge 
        tX.draw()
        tY = line(0,0,0,0,1,0,0,1,0,self.program) #création d'un ligne en Y couleur vert
        tY.draw()
        tZ = line(0,0,0,0,0,1,0,0,1,self.program) #création d'un ligne en Z couleur bleue
        tZ.draw()

    def on_draw(self, event):
        gloo.set_clear_color('grey')
        gloo.clear(color=True)
        self.drawFrame()

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size) #l'écran d'affichage

    def activate_zoom(self): #pour crer la matrice de projection
        gloo.set_viewport(0, 0, *self.physical_size)
        projection = perspective(45.0, self.size[0] / float(self.size[1]),2.0, 10.0) #matrice de projection
        self.program['projection'] = projection

    #def on_timer(self, event):
    #    self.thetax = self.thetax+1
    #    self.program['model'] = rotate(self.thetax, (1, 0, 0))
    #    self.update() # on remet à jour et on redessine

    def on_mouse_move(self,event):
        self.new_method(event)
        x = event.pos[0] 
        y = event.pos[1] 
        #print(self.size)               #use to know the size of the canva
        #print (event.pos)              #use to verify position reading work well
        print()
        thetaX = ((self.size[0]/2)/self.size[1])*y -self.size[0]/2
        thetaY = ((self.size[1]/2)/self.size[0])*x -self.size[1]/2
        Rx = rotate(thetaX, (1,0,0))
        Ry = rotate(thetaY, (0,1,0))
        R = matmul(Rx, Ry)

        self.program['model'] = R
        self.update() # on remet à jour et on redessine

       
        #self.thetax = self.mathmul(-180, 180)
        #self.program['model'] = rotate(self.thetax, (1, 0, 0))

    def new_method(self, event):
        x = (event.pos[0])




if __name__ == "__main__":
    c = Canvas() #construction d'un objet Canvas
    app.run()