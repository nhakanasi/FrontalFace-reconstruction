import os ; import sys 
from show3d.chj.ogl import *
from show3d.chj.ogl.objloader import CHJ_tiny_obj
from show3d.chj.ogl import light
import time

class param(object):pass

def show_3dface(fobj):
    obj = CHJ_tiny_obj( "", fobj, swapyz=False)
    obj.create_bbox()
    
    param.obj=obj
    param.sel_pos = False

    pygame.init()
    viewport = (800, 800)
    param.viewport = viewport
    # screen = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)
    screen = pygame.display.set_mode(viewport, pygame.DOUBLEBUF | pygame.OPENGL)

    param.screen = screen

    light.setup_lighting()
    glLightfv(GL_LIGHT0, GL_POSITION, (0, 0, 1000, 0))

    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)

    obj.create_gl_list()

    clock = pygame.time.Clock()

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    cam = light.camera
    cam.Ortho.bbox[:] = cam.Ortho.bbox * 13
    cam.Ortho.nf[:] = cam.Ortho.nf * 200
    gluPerspective(60, 1, 0.1, 10000)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0, 0, 800, 0, 0, 0, 0, 1, 0)

    rx, ry = (0, 0)
    tx, ty = (0, 0)
    zpos = 5
    rotate = move = False
    running = True
    try:
        while running:
            clock.tick(30)
            
            for e in pygame.event.get():
                if e.type == QUIT:
                    pygame.quit()
                    running = False
                    time.sleep(0.5)
                    # sys.exit(2)
                if e.type == KEYDOWN:
                    if e.key == pygame.K_ESCAPE:
                        pygame.quit()
                        running = False
                        time.sleep(0.5)
                        # sys.exit(2)
                        
                    elif e.key == pygame.K_4:
                        param.sel_pos = not param.sel_pos

                elif e.type == MOUSEBUTTONDOWN:

                    pressed_array = pygame.mouse.get_pressed()
                    if pressed_array[0]:
                        if param.sel_pos:
                            pos = pygame.mouse.get_pos()
                            pos_get_pos3d_show(pos)

                    if e.button == 4:
                        zpos = max(1, zpos - 1)
                    elif e.button == 5:
                        zpos += 1
                    elif e.button == 1:
                        rotate = True
                    elif e.button == 3:
                        move = True
                elif e.type == MOUSEBUTTONUP:
                    if e.button == 1:
                        rotate = False
                    elif e.button == 3:
                        move = False
                elif e.type == MOUSEMOTION:
                    i, j = e.rel
                    if rotate:
                        rx -= i
                        ry -= j
                    if move:
                        tx += i
                        ty -= j
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            glLoadIdentity()
            glTranslate(tx / 20., ty / 20., - zpos)
            glRotate(ry / 5, 1, 0, 0)
            glRotate(rx / 5, 0, 1, 0)
            s = [2 / obj.bbox_half_r] * 3
            glScale(*s)

            t = -obj.bbox_center
            glTranslate(*t)

            glCallList(obj.gl_list)
            if hasattr(param, 'pos3d') and param.sel_pos:
                draw_pos(param.pos3d)
            if running:
                pygame.display.flip()
    except:
        pygame.quit()


def pos_get_pos3d(pos):
    x = pos[0]
    y = param.viewport[1] - pos[1]
    z = glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT )
    modelview=glGetDoublev(GL_MODELVIEW_MATRIX)
    projection=glGetDoublev(GL_PROJECTION_MATRIX )
    viewport=glGetIntegerv(GL_VIEWPORT)
    xyz=gluUnProject(x,y,z, modelview, projection, viewport)
    return xyz

def pos_get_pos3d_show(pos):
    pos3d=pos_get_pos3d(pos)
    param.pos3d=pos3d
    p("pos3d",pos3d)

def draw_pos(pos3d, size=10, color=[0,1,0]):
    glPointSize(size)
    glBegin(GL_POINTS)

    glColor3f(*color)
    glVertex3f(*pos3d)

    glEnd()
    glColor3f(1, 1, 1)