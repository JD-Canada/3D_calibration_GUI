from PyQt5.QtWidgets import (QMainWindow, QApplication, QSlider,QColorDialog,
        QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsScene, QTableWidget,QTableWidgetItem, QVBoxLayout,
        QGraphicsItem, QGraphicsView, QVBoxLayout, QGridLayout,QGraphicsPixmapItem, QFrame,
        QPushButton,QTableView, QGraphicsItemGroup, QLabel, QFileDialog, QInputDialog, QLineEdit, QMessageBox, QGraphicsSimpleTextItem)
from PyQt5.QtGui import QPainter, QTransform, QColor, QPixmap, QStandardItemModel, QStandardItem, QIcon, QBrush
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot,QPointF, QPoint, QRectF, QAbstractTableModel

import sys
import csv
import os
import cv2
import numpy as N
import pandas as pd
import time

import user_interface

"""
class ImageViewer(QGraphicsView):
    photoClicked = pyqtSignal(QPoint)

    def __init__(self, parent):
        super(ImageViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QGraphicsScene(self)
        self._photo = QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setFrameShape(QFrame.NoFrame)

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QGraphicsView.NoDrag)
            self._photo.setPixmap(QPixmap())
        self.fitInView()

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QGraphicsView.ScrollHandDrag:
            self.setDragMode(QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(ImageViewer, self).mousePressEvent(event)
"""

class PandasModel(QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """
    def __init__(self, data, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.values[index.row()][index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None

class Image(QGraphicsPixmapItem):
    """ create Pixmap image for each scene"""
    
    def __init__(self,path):
        super().__init__()
        
        self.setPixmap(QPixmap(path))

class Scene(QGraphicsScene):
    entered = pyqtSignal([QGraphicsItem],[QGraphicsEllipseItem])
    leave = pyqtSignal([QGraphicsItem],[QGraphicsEllipseItem])
    move = pyqtSignal([QGraphicsItem],[QGraphicsEllipseItem])
    # press = pyqtSignal([QGraphicsItem],[QGraphicsEllipseItem])
    """Each instance holds all image and ellipse data"""
    def __init__(self,MainWindow,**keywords):
        # executes QGraphicsScene.__init__()
        super().__init__()
        
        # self.sceneCount=self.counter
        self.MainWindow=MainWindow
        self.keywords=keywords
        self.threshold=self.MainWindow.binary_sb.value()
        self.populated=False

        if 'path' in self.keywords.keys():
            self.getPathData(self.keywords['path'])
            self.binarize_image()
            self.findContours()
            self.image = Image(self.path)   
            self.addItem(self.image)

        if 'headers' in self.keywords.keys():
            self.headx_px = self.keywords['headers'][0]
            self.heady_px = self.keywords['headers'][1]
            xheader = self.headx_px.split("_")
            yheader = self.heady_px.split("_")
            self.sceneID = list(set(xheader).intersection(set(yheader)))[0]
        else:
            self.headx_px="%s_x" %self.filename
            self.heady_px="%s_y" %self.filename
            self.sceneID="%s" %(self.filename)

        self.dotCoords={}
        if 'coords' in self.keywords.keys():
            self.pointCount=len(self.keywords['coords'][0])
            self.dotCoords['x_px']= self.keywords['coords'][0]
            self.dotCoords['y_px']= self.keywords['coords'][1]
        else:
            self.pointCount=0

            self.dotCoords['x_px']=[None]*len(self.MainWindow.pointData.index)
            self.dotCoords['y_px']=[None]*len(self.MainWindow.pointData.index)

        if self.sceneID in self.MainWindow.scenes:
            self.sceneID="%s_copy" %(self.sceneID)
        
        self.MainWindow.view_lw.addItem(self.sceneID)

    def setThreshold(self):
        self.threshold=self.MainWindow.binary_sb.value()
        
    def populateDotsOnLink(self):
        
        self.pointCount=0
        for i in range(len(self.MainWindow.pointData.index)):
            self.circle=Ellipse(0,0, 10, 10,self.pointCount)
            self.circle.setPos(self.dotCoords['x_px'][i]-5,self.dotCoords['y_px'][i]-5)
            self.addItem(self.circle)
            self.pointCount += 1
        self.populated=True

    def linkView(self,path):

        if path[0]!='':

            self.getPathData(path[0])
            
            
            self.clear()
            
            self.image = Image(self.path)   
            self.addItem(self.image)
            self.binarize_image()
            self.findContours()

            self.populateDotsOnLink()

    def getPathData(self,path):
        self.path=path
        self.filename_w_ext = os.path.basename(path)
        self.filename, file_extension = os.path.splitext(self.filename_w_ext)
        self.root=os.path.dirname(os.path.abspath(path))

    def remove_image(self):
        self.removeItem(self.image)
    
    def refresh_image(self,path):
        self.image = Image(self.path)        
        self.addItem(self.image)

    def binarize_image(self):
        
        self.orig_image = cv2.imread(self.path)
        self.cnts_image=self.orig_image
        imageGray = cv2.cvtColor(self.orig_image,cv2.COLOR_BGR2GRAY)
        
        ret,self.cvBinaryImage = cv2.threshold(imageGray,self.threshold,255,cv2.THRESH_BINARY_INV)

        self.binaryImagePath=self.root+"\\" + self.filename+'_binary.jpg'
        cv2.imwrite(self.binaryImagePath, self.cvBinaryImage)
        
        # self.binarySceneName='%s_binary' % str(self.sceneID)
        self.binaryImage = Image(self.binaryImagePath)  
        
    def findContours(self):

        #Prevents a fatal crash due to version conflict in cv2.findContours
        try:
            
            self.cnts, hierachy = cv2.findContours(self.cvBinaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            ret, self.cnts, hierachy = cv2.findContours(self.cvBinaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)           
    
        self.good_cnts=[]
        for c in self.cnts:
            
            if 5 <= cv2.contourArea(c) <= 500:
                self.good_cnts.append(c)
                cv2.drawContours(self.cnts_image, [c], -5, (255, 0, 0), 1)
                
                (xstart, ystart, w, h) = cv2.boundingRect(c)
                M = cv2.moments(c)
                try:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    cv2.circle(self.cnts_image,(cx,cy),2,(0,0,255),-1)
                except ZeroDivisionError:
                    continue
        
        self.cntsImagePath=self.root+"\\" + self.filename+'_cnts.jpg'
        cv2.imwrite(self.cntsImagePath, self.cnts_image)
        self.contourImage=Image(self.cntsImagePath)

    def clearView(self):
        self.removeItem(self.image)
        self.removeItem(self.binaryImage)
        self.removeItem(self.contourImage)

    def showBinaryImage(self):
        self.clearView()
        self.addItem(self.binaryImage)

    def showContourImage(self):
        self.clearView()
        self.addItem(self.contourImage)

    def showOriginalImage(self):
        self.clearView()
        self.addItem(self.image)

    def correct_centers(self):
        
        self.findContours()

        for idx, i in enumerate(self.items()):
            
            if isinstance(i, Ellipse):
                
                for c in self.good_cnts:
                    (xstart, ystart, w, h) = cv2.boundingRect(c)

                    x=i.pos().toPoint().x()+5
                    y=i.pos().toPoint().y()+5
                    
                    if xstart <= x <= xstart+w and ystart <= y <= ystart+h:
                        M = cv2.moments(c)
                        try:
                            cx = int(M['m10']/M['m00'])
                            cy = int(M['m01']/M['m00'])

                            i.setX(cx-5)
                            i.setY(cy-5)
                            self.MainWindow.ellipseMove(i)
                            i.setBrush(QColor('green'))
                            
                        except ZeroDivisionError:
                            continue

    def showCoords(self, event):
        try:
            self.removeItem(self.text)
            self.removeItem(self.dot)
        except AttributeError:
            pass
        p=self.MainWindow.View.mapToScene(event.x(),event.y())
        self.dot=QGraphicsEllipseItem(int(p.x())-3,int(p.y())-3, 6, 6)
        self.dot.setBrush(Qt.blue)
        self.addItem(self.dot)
        self.text=QGraphicsSimpleTextItem("x%s_y%s" % (str(int(p.x())+5),str(int(p.y())+5)))
        self.text.setBrush(Qt.red)
        self.text.setPos(p.x(), p.y())
        self.addItem(self.text)
        # self.removeItem(self.text)

    def removeInfo(self, item):
        self.current_scene.removeItem(self.text)

    def add_point(self, event):
        
        try:
            self.path
        except AttributeError:
            self.MainWindow.errMessage="Please link calibration point data with a calibration view"
            self.MainWindow.errorTitle="Calibration view not linked"
            self.MainWindow.errorMsg()
            return None             

        if self.pointCount == self.MainWindow.num_points:
            self.MainWindow.errMessage="You have placed all the points listed in the calibration point file. If you wish to add more points, add more rows to the point file and start over."
            self.MainWindow.errorTitle="Cannot place any more points"
            self.MainWindow.errorMsg()
            return None

        p=self.MainWindow.View.mapToScene(event.x(),event.y())
        self.circle=Ellipse(0,0, 10, 10,self.pointCount)
        self.circle.setPos(p.x()-5,p.y()-5)
        self.addItem(self.circle)
        self.dotCoords['x_px'][self.pointCount]=int(p.x())
        self.dotCoords['y_px'][self.pointCount]=int(p.y())
        self.MainWindow.refreshTableData()
        self.pointCount += 1

    def updateTableOnEllipseMove(self,x,y,item):
        
        self.dotCoords['x_px'][item.count]=int(x)
        self.dotCoords['y_px'][item.count]=int(y)
        self.MainWindow.refreshTableData()

class Ellipse(QGraphicsEllipseItem):
    """creates circles placed in scene"""
    moved = pyqtSignal(int,int)
    def __init__(self, x, y, w, h,count):
        super().__init__(x, y, w, h)

        self.count=count
        self.setBrush(QColor('red'))
        self.setPen(QColor('black'))
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.SizeAllCursor)
        self.setZValue(100)

    def hoverEnterEvent(self, event) :
        super().hoverEnterEvent(event)
        self.scene().entered.emit(self)
        self.update()
        
    def hoverLeaveEvent(self, event):
        super().hoverEnterEvent(event)
        self.scene().leave.emit(self)
        self.update()
      
    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        self.scene().move.emit(self)
        self.update()

class DLT:

    def __init__(self,nd,nc):
        
        self.nd=nd
        self.nc=nc

    def DLTcalib(self, nd, xyz, uv):
        """
        Camera calibration by DLT using known object points and their image points.

        This code performs 2D or 3D DLT camera calibration with any number of views (cameras).
        For 3D DLT, at least two views (cameras) are necessary.
        Inputs:
        nd is the number of dimensions of the object space: 3 for 3D DLT and 2 for 2D DLT.
        xyz are the coordinates in the object 3D or 2D space of the calibration points.
        uv are the coordinates in the image 2D space of these calibration points.
        The coordinates (x,y,z and u,v) are given as columns and the different points as rows.
        For the 2D DLT (object planar space), only the first 2 columns (x and y) are used.
        There must be at least 6 calibration points for the 3D DLT and 4 for the 2D DLT.
        Outputs:
        L: array of the 8 or 11 parameters of the calibration matrix
        err: error of the DLT (mean residual of the DLT transformation in units of camera coordinates).
        """
        #Convert all variables to numpy array:
        xyz = N.asarray(xyz)
        uv = N.asarray(uv)
        #number of points:
        np = xyz.shape[0]
        #Check the parameters:
        if uv.shape[0] != np:
            raise ValueError('xyz (%d points) and uv (%d points) have different number of points.' %(np, uv.shape[0]))
        if (nd == 2 and xyz.shape[1] != 2) or (nd == 3 and xyz.shape[1] != 3):
            raise ValueError('Incorrect number of coordinates (%d) for %dD DLT (it should be %d).' %(xyz.shape[1],nd,nd))
        if nd == 3 and np < 6 or nd == 2 and np < 4:
            raise ValueError('%dD DLT requires at least %d calibration points. Only %d points were entered.' %(nd, 2*nd, np))
            
        #Normalize the data to improve the DLT quality (DLT is dependent of the system of coordinates).
        #This is relevant when there is a considerable perspective distortionp.
        #Normalization: mean position at origin and mean distance equals to 1 at each directionp.
        Txyz, xyzn = self.Normalization(nd, xyz)
        Tuv, uvn = self.Normalization(2, uv)

        A = []
        if nd == 2: #2D DLT
            for i in range(np):
                x,y = xyzn[i,0], xyzn[i,1]
                u,v = uvn[i,0], uvn[i,1]
                A.append( [x, y, 1, 0, 0, 0, -u*x, -u*y, -u] )
                A.append( [0, 0, 0, x, y, 1, -v*x, -v*y, -v] )
        elif nd == 3: #3D DLT
            for i in range(np):
                x,y,z = xyzn[i,0], xyzn[i,1], xyzn[i,2]
                u,v = uvn[i,0], uvn[i,1]
                A.append( [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u] )
                A.append( [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v] )

        #convert A to array
        A = N.asarray(A) 
        #Find the 11 (or 8 for 2D DLT) parameters:
        U, S, Vh = N.linalg.svd(A)
        #The parameters are in the last line of Vh and normalize them:
        L = Vh[-1,:] / Vh[-1,-1]
        #Camera projection matrix:
        H = L.reshape(3,nd+1)
        #Denormalization:
        H = N.dot( N.dot( N.linalg.pinv(Tuv), H ), Txyz );
        H = H / H[-1,-1]
        self.L = H.flatten(0)
        #Mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates):
        uv2 = N.dot( H, N.concatenate( (xyz.T, N.ones((1,xyz.shape[0]))) ) ) 
        uv2 = uv2/uv2[2,:] 
        #mean distance:
        self.err = N.sqrt( N.mean(N.sum( (uv2[0:2,:].T - uv)**2,1 )) ) 

        return self.L, self.err

    def Normalization(self,nd,x):
        '''
        Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3).

        Inputs:
            nd: number of dimensions (2 for 2D; 3 for 3D)
            x: the data to be normalized (directions at different columns and points at rows)
        Outputs:
            Tr: the transformation matrix (translation plus scaling)
            x: the transformed data
        '''

        x = N.asarray(x)
        m, s = N.mean(x,0), N.std(x)
        if nd==2:
            Tr = N.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
        else:
            Tr = N.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])
            
        Tr = N.linalg.inv(Tr)
        x = N.dot( Tr, N.concatenate( (x.T, N.ones((1,x.shape[0]))) ) )
        x = x[0:nd,:].T

        return Tr, x

    def DLTrecon(self,nd, nc, Ls, uvs):
        '''
        Reconstruction of object point from image point(s) based on the DLT parameters.

        This code performs 2D or 3D DLT point reconstruction with any number of views (cameras).
        For 3D DLT, at least two views (cameras) are necessary.
        Inputs:
        nd is the number of dimensions of the object space: 3 for 3D DLT and 2 for 2D DLT.
        nc is the number of cameras (views) used.
        Ls (array type) are the camera calibration parameters of each camera 
        (is the output of DLTcalib function). The Ls parameters are given as columns
        and the Ls for different cameras as rows.
        uvs are the coordinates of the point in the image 2D space of each camera.
        The coordinates of the point are given as columns and the different views as rows.
        Outputs:
        xyz: point coordinates in space
        '''
        
        #Convert Ls to array:
        Ls = N.asarray(Ls) 
        #Check the parameters:
        if Ls.ndim ==1 and nc != 1:
            raise ValueError('Number of views (%d) and number of sets of camera calibration parameters (1) are different.' %(nc))    
        if Ls.ndim > 1 and nc != Ls.shape[0]:
            raise ValueError('Number of views (%d) and number of sets of camera calibration parameters (%d) are different.' %(nc, Ls.shape[0]))
        if nd == 3 and Ls.ndim == 1:
            raise ValueError('At least two sets of camera calibration parameters are needed for 3D point reconstructionp.')

        if nc == 1: #2D and 1 camera (view), the simplest (and fastest) case
            #One could calculate inv(H) and input that to the code to speed up things if needed.
            #(If there is only 1 camera, this transformation is all Floatcanvas2 might need)
            Hinv = N.linalg.inv( Ls.reshape(3,3) )
            #Point coordinates in space:
            xyz = N.dot( Hinv,[uvs[0],uvs[1],1] ) 
            xyz = xyz[0:2]/xyz[2]       
        else:
            M = []
            for i in range(nc):
                L = Ls[i,:]
                u,v = uvs[i][0], uvs[i][1] #this indexing works for both list and numpy array
                if nd == 2:      
                    M.append( [L[0]-u*L[6], L[1]-u*L[7], L[2]-u*L[8]] )
                    M.append( [L[3]-v*L[6], L[4]-v*L[7], L[5]-v*L[8]] )
                elif nd == 3:  
                    M.append( [L[0]-u*L[8], L[1]-u*L[9], L[2]-u*L[10], L[3]-u*L[11]] )
                    M.append( [L[4]-v*L[8], L[5]-v*L[9], L[6]-v*L[10], L[7]-v*L[11]] )
            
            #Find the xyz coordinates:
            U, S, Vh = N.linalg.svd(N.asarray(M))
            #Point coordinates in space:
            xyz = Vh[-1,0:-1] / Vh[-1,-1]
        
        return xyz

class MainWindow(QMainWindow, user_interface.Ui_MainWindow):

    """main user interface """
    def __init__(self, parent=None):
        
        super(MainWindow, self).__init__(parent)

        self.setupUi(self)
        self.setWindowTitle("Calibrate 3D")
        self.setWindowIcon(QIcon(os.path.dirname(os.path.abspath(__file__))+'/dice.png'))
        
        self.scenes={}
        self.calibrate=DLT(3,2)
        self.tableCreated=False

        self.initUI()

    def initUI(self):   
        
        #button connects
        self.addPointFile_b.clicked.connect(self.newGetCalibrationPoints)
        self.loadFilledTable_b.clicked.connect(self.readInTable)
        self.loadCal_b.clicked.connect(self.loadCalibration)
        self.newView_b.clicked.connect(self.new_add_scene)
        self.link_b.clicked.connect(self.linkView)
        self.correct_b.clicked.connect(self.button_correct_centers)
        self.binary_sb.valueChanged.connect(self.redo_binarization)
        self.saveTable_b.clicked.connect(self.saveTable)
        self.calibrate_b.clicked.connect(self.calibrate3D)
        self.loadTest_b.clicked.connect(self.loadTestTableData)
        self.testCal_b.clicked.connect(self.test3DCalibration)
        self.delete_b.clicked.connect(self.deleteView)
        self.saveTest_b.clicked.connect(self.saveTestData)
        self.saveCalibration_b.clicked.connect(self.saveCalibration)
        
        #list widget connects
        # self.view_lw.clicked.connect(self.change_scene)
        self.view_lw.currentRowChanged.connect(self.change_scene)
        self.original_rb.toggled.connect(self.change_scene)
        self.binary_rb.toggled.connect(self.change_scene)
        self.contour_rb.toggled.connect(self.change_scene)
        
        #other
        self.View.mouseDoubleClickEvent= self.add_point
        self.slider.setRange(1, 500)
        self.slider.setValue(100)
        self.slider.valueChanged[int].connect(self.onZoom)
        self.groupBox_8.setVisible(False)
        self.link_b.setVisible(False)
        self.radioButton_2.toggled['bool'].connect(self.link_b.setVisible)

    def linkView(self):

        if self.view_lw.count() == 0:
            self.errMessage="No views available for linking."
            self.errorTitle="Cannot link view"
            self.errorMsg()
            return None            

        self.selectedItemText=self.view_lw.currentItem().text()
        self.current_scene=self.scenes[self.selectedItemText]

        path=QFileDialog.getOpenFileName(self,
        "Load image view",filter="Image Files( *.png *.jpg)")

        self.scenes[self.selectedItemText].linkView(path)
        self.change_scene()
        self.View.setScene(self.current_scene)

    def deleteView(self):
        
        try:
            self.selectedItemText=self.view_lw.currentItem().text()
        except AttributeError:
            self.errMessage="Please click on the view to remove."
            self.errorTitle="No view selected"
            self.errorMsg()
            return None
        xCol=str(self.scenes[self.selectedItemText].headx_px)
        yCol=str(self.scenes[self.selectedItemText].heady_px)
        self.pointData.drop(columns=[xCol, yCol],inplace =True)
        self.scenes.pop(self.selectedItemText)
        self.view_lw.takeItem(self.view_lw.currentRow())

        try:
            self.firstSceneKey=list(self.scenes.keys())[0]
            self.current_scene=self.scenes[self.firstSceneKey]
            self.current_scene.showOriginalImage()
            item=self.view_lw.findItems(self.current_scene.sceneID,Qt.MatchExactly)[0]
            item.setSelected(True)
            self.view_lw.setCurrentItem(item)
            self.View.setScene(self.current_scene)
        except IndexError:
            self.current_scene.clear()
        except AttributeError:
            pass
        
        self.tableWidget.setModel(PandasModel(self.pointData))

    """
    Table loading and manipulation methods
    """
    def newGetCalibrationPoints(self):

        self.promptSceneDelete()
        pointFilePath=QFileDialog.getOpenFileName(self,"Load point data file",filter="Text files( *.txt *.csv)")

        if pointFilePath[0]!='':

            self.pointData=pd.read_csv(pointFilePath[0])
            self.num_points=len(self.pointData.index)
            self.nmbViews=(len(self.pointData.columns)-3)/2
            if self.nmbViews > 0:
                self.errMessage="Please load a file containing colums for only the point coordinates."
                self.errorTitle="File contains view data!"
                self.errorMsg()
                return None  
            self.tableWidget.setModel(PandasModel(self.pointData))
            self.tabWidget.setCurrentIndex(1)
            self.pointFile_l.setText(pointFilePath[0])         
    
    def refreshTableData(self):

        for key in self.scenes:

            for inner_key in self.scenes[key].dotCoords:
                if inner_key == 'x_px':
                    self.pointData[self.scenes[key].headx_px]=self.scenes[key].dotCoords[inner_key]
                if inner_key == 'y_px':
                    self.pointData[self.scenes[key].heady_px]=self.scenes[key].dotCoords[inner_key]

        self.tableWidget.setModel(PandasModel(self.pointData))

    def promptSceneDelete(self):

        if self.view_lw.count() > 0:

            reply = QMessageBox.question(self, 'Continue?', 
                    'Loading a new calibration table will delete existing views, accept?', QMessageBox.Yes, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.scenes={}
                self.view_lw.clear()
            else:
                return None
            
    
    def readInTable(self):

        self.promptSceneDelete()
        pointFilePath=QFileDialog.getOpenFileName(self,"Load populated point data file",filter="Text files( *.txt *.csv)")

        if pointFilePath[0]!='':
            self.pointData=pd.read_csv(pointFilePath[0])
            self.num_points=len(self.pointData.index)
            self.nmbViews=int((len(self.pointData.columns)-3)/2)
            self.headers=list(self.pointData.columns) 

            if self.nmbViews < 2:
                self.errMessage="File does not contain data sufficient for at least two views"
                self.errorTitle="Not enough views in file"
                self.errorMsg()
                return None 

            col=3
            for i in range(self.nmbViews):
                colNames=[]
                colNames.append(self.headers[col])
                colNames.append(self.headers[col+1])
                
                coords=[]
                coords.append(self.pointData[self.headers[col]])
                coords.append(self.pointData[self.headers[col+1]])

                self.current_scene=Scene(self,coords=coords,headers=colNames)
                self.current_scene.entered.connect(self.displayInfo)
                self.current_scene.leave.connect(self.removeInfo)
                self.current_scene.move.connect(self.ellipseMove)
                
                self.scenes[self.current_scene.sceneID]=self.current_scene
                # self.View.setScene(self.current_scene)
                item=self.view_lw.findItems(self.current_scene.sceneID,Qt.MatchExactly)[0]
                item.setSelected(True)
                self.view_lw.setCurrentItem(item)
                # self.newView_b.setText("Add view")
                
                col+=2

            self.pointFile_l.setText(pointFilePath[0])
            self.tableWidget.setModel(PandasModel(self.pointData))
            self.tabWidget.setCurrentIndex(1)
            
    def saveTable(self):

        path = QFileDialog.getSaveFileName(
                self, 'Save File', '', 'CSV(*.csv)')[0]
        self.pointData.to_csv(path,index=False)
        

    """
    View and Scene related methods
    """

    def new_add_scene(self):

        if not hasattr(self, 'pointData'):

            self.errMessage="Please load a point file or a populate calibration file and then try again."
            self.errorTitle="Cannot add view!"
            self.errorMsg()
            return None       

        viewPath=QFileDialog.getOpenFileName(self,
        "Load image view",filter="Image Files( *.png *.jpg)")

        if viewPath[0]!='':

            self.current_scene=Scene(self,path=viewPath[0])
            self.current_scene.entered.connect(self.displayInfo)
            self.current_scene.leave.connect(self.removeInfo)
            self.current_scene.move.connect(self.ellipseMove)
            
            self.scenes[self.current_scene.sceneID]=self.current_scene
            self.View.setScene(self.current_scene)
            item=self.view_lw.findItems(self.current_scene.sceneID,Qt.MatchExactly)[0]
            item.setSelected(True)
            self.view_lw.setCurrentItem(item)
            self.newView_b.setText("Add view")

            self.pointData[self.current_scene.headx_px]=N.nan
            self.pointData[self.current_scene.heady_px]=N.nan
            self.tableWidget.setModel(PandasModel(self.pointData))

    def my_decorator(self):
        
        for count, scene in enumerate(self.scenes):
            
            if scene in self.selectedItemText:
                
                self.current_scene=self.scenes[scene]
                

    def change_scene(self):
        self.tabWidget.setCurrentIndex(0)
        try:
            self.selectedItemText=self.view_lw.currentItem().text()
           
            self.my_decorator()

            if not hasattr(self.current_scene, 'path'):
                emptyScene=QGraphicsScene()
                self.View.setScene(emptyScene)
                return None
                
            self.binary_sb.setValue(self.current_scene.threshold)
            
            if self.original_rb.isChecked()==True:
                self.current_scene.showOriginalImage()
            elif self.binary_rb.isChecked()==True:
                self.current_scene.showBinaryImage()
            elif self.contour_rb.isChecked()==True:
                self.current_scene.showContourImage()
            self.View.setScene(self.current_scene)
        except AttributeError:
            pass

    def redo_binarization(self):

        if not hasattr(self, 'pointData'):
                self.errMessage="Please select a view then try again."
                self.errorTitle="No view selected!"
                self.errorMsg()
                return None    

        try:
            self.selectedItemText=self.view_lw.currentItem().text()
            
        except IndexError:
            self.errMessage="Please click on a view to work with."
            self.errorTitle="No view selected"
            self.errorMsg()
            return None
        except AttributeError:
            self.errMessage="Please click on a view to work with."
            self.errorTitle="No view selected"
            self.errorMsg()
            return None      

        try:
            self.my_decorator()
            self.current_scene.setThreshold()
            self.current_scene.findContours()
            self.current_scene.binarize_image()
            self.change_scene()
        except AttributeError:
            pass

    def onZoom(self, value):
        val = value / 100    
        self.View.resetTransform()
        self.View.scale(val, val)    

    """
    Calibration marker methods
    """
    def add_point(self, event):
        self.current_scene.add_point(event)

    def showCoords(self, event):
        self.current_scene.showCoords(event)

    def button_correct_centers(self):
        try:
            self.current_scene.correct_centers()
            self.change_scene()
        except AttributeError:
            pass

    def displayInfo(self, item):
        self.text=QGraphicsSimpleTextItem("P%s_x%s_y%s" % (str(item.count+1),str(item.pos().toPoint().x()+5),str(item.pos().toPoint().y()+5)))
        self.text.setBrush(Qt.red)
        self.text.setPos(item.pos().toPoint().x()+10, item.pos().toPoint().y()+10)
        self.current_scene.addItem(self.text)

    def removeInfo(self, item):
        self.current_scene.removeItem(self.text)

    def ellipseMove(self, item):
        item.setBrush(QColor('red'))
        self.current_scene.removeItem(self.text)
        x=str(item.pos().toPoint().x()+5)
        y=str(item.pos().toPoint().y()+5)  

        self.current_scene.updateTableOnEllipseMove(x,y,item)         

    """
    Calibration and testing methods
    """
    def calibrate3D(self):

        try:
            x=list(self.pointData['x'])
            y=list(self.pointData['y'])
            z=list(self.pointData['z'])

        except AttributeError:
            self.errMessage="Fully populate the calibration table to continue."
            self.errorTitle="Calibration table not correctly populated!"
            self.errorMsg()
            return None   

        #set up xyz coords list
        xyz=[]

        for i in range(len(x)):
                point=[]
                point.append(x[i])
                point.append(y[i])
                point.append(z[i])
                print(point)
                xyz.append(point)
 
        px_coords=[]
        for key in self.scenes:
            view_coords=[]
            x_px=list(self.pointData[self.scenes[key].headx_px])
            y_px=list(self.pointData[self.scenes[key].heady_px])
            view_coords.append(x_px)
            view_coords.append(y_px)
            px_coords.append(view_coords)
        
        self.nmbDim=3
        self.coefficients=[]
        errors=[]
        
        self.nmbCam=self.nmbViews
    
        #get parameters for each view
        for view in px_coords:
            uv=[]
            for i in range(len(view[0])):
                
                pair=[]
                pair.append(view[0][i])
                pair.append(view[1][i])
                uv.append(pair)
            L, err = self.calibrate.DLTcalib(self.nmbDim, xyz, uv)
            
            self.coefficients.append(L)
            errors.append(err)
        

    def loadTestTableData(self):

        pointFilePath=QFileDialog.getOpenFileName(self,"Load populated test table",filter="Text files( *.txt *.csv)")

        if pointFilePath[0]!='':
            self.testData=pd.read_csv(pointFilePath[0])
            self.numTestPoints=len(self.testData.index)
            self.numTestViews=int((len(self.testData.columns)-3)/2)
            self.testHeaders=list(self.testData.columns) 

            if self.numTestViews < 2:
                self.errMessage="File contains less than two views"
                self.errorTitle="Not enough views in file"
                self.errorMsg()
                return None 

        self.test_tableView.setModel(PandasModel(self.testData))

    def saveCalibration(self):

        try:
            test=self.coefficients

        except AttributeError:
            self.errMessage="Perform a calibration and then try again."
            self.errorTitle="No calibration available!"
            self.errorMsg()
            return None   

        path = QFileDialog.getSaveFileName(
                self, 'Save File', '', 'CSV(*.csv)')[0]
        if path:
            print(self.coefficients)
            self.coefficients_df=pd.DataFrame(self.coefficients)
            self.coefficients_df.to_csv(path,sep=',',index=False,header=False)

    def loadCalibration(self):
        pointFilePath=QFileDialog.getOpenFileName(self,"Load calibration file",filter="Text files( *.txt *.csv)")
        self.coefficients=[]
    
        if pointFilePath[0]!='':
            self.dfCoefficients=pd.read_csv(pointFilePath[0],header=None)
        
            for index, row in self.dfCoefficients.iterrows():
                self.coefficients.append(N.asarray(list(row)))
        self.tabWidget.setCurrentIndex(2)

    def test3DCalibration(self):            

            x=list(self.testData[self.testHeaders[0]])
            y=list(self.testData[self.testHeaders[1]])
            z=list(self.testData[self.testHeaders[2]])

            self.testXYZ=[]
            for i in range(self.numTestPoints):
                testCoord=[]
                testCoord.append(x[i])
                testCoord.append(y[i])
                testCoord.append(z[i])
                self.testXYZ.append(testCoord)
            
            testPxCoords=[]
            col=3
            for i in range(self.numTestViews):
                colNames=[]

                colNames.append(self.testHeaders[col])
                colNames.append(self.testHeaders[col+1])
                x_px=list(self.testData[self.testHeaders[col]])
                y_px=list(self.testData[self.testHeaders[col+1]])
            
                coords=[]
                for i in range(len(x_px)):
                    pair=[]
                    pair.append(x_px[i])
                    pair.append(y_px[i])
                    coords.append(pair)

                testPxCoords.append(coords)
                col+=2


            xyz1234 = N.zeros((len(self.testXYZ),3))
            print(self.coefficients)
            #use parameters to reconstruct input points 
            for i in range(len(testPxCoords[0])):
                i_px_coords=[]
                for view in testPxCoords:
                    i_px_coords.append(view[i])
                try:
                    xyz1234[i,:] = self.calibrate.DLTrecon(3,self.numTestViews,self.coefficients,i_px_coords)
                except AttributeError:
                    self.errMessage="No calibration available"
                    self.errorTitle="Perform a calibration, then try again."
                    self.errorMsg()
                    return None   
                except ValueError:
                    self.errMessage="Ensure you have loaded a valid calibration file and a valid marker test point file. The calibration file should have 12 columns and as many rows as were used to perform the calibration. The test point file should have 2x the number of views used to obtain the calibration plus 3 more columns for the object-space coordinates."
                    self.errorTitle="Calibration and test point files are incompatabile!"
                    self.errorMsg()
                    return None                                       


            rec_xyz=pd.DataFrame(xyz1234,columns=['rec_x','rec_y','rec_z'])
            xyz=pd.DataFrame(self.testXYZ,columns=['x','y','z'])
            self.results=pd.concat([xyz,rec_xyz], axis=1)
            self.results['x_e']=self.results['x']-self.results['rec_x']
            self.results['y_e']=self.results['y']-self.results['rec_y']
            self.results['z_e']=self.results['z']-self.results['rec_z']

            self.xmean_l.setText("%.4f" % ((self.results.x - self.results.rec_x)**2).mean()**.5)
            self.ymean_l.setText("%.4f" % ((self.results.y - self.results.rec_y)**2).mean()**.5)
            self.zmean_l.setText("%.4f" % ((self.results.z - self.results.rec_z)**2).mean()**.5)

            self.pointsTested_l.setText(str(self.numTestPoints))

            self.errorTitle="Calibration test successful!"
            self.errMessage="The test completed with success. If desired, consult the simple statistics below or save the test's results and postprocess with a third party application."
            self.errorMsg()
            # print(self.results)

            # print(N.mean(N.sqrt(N.sum((N.array(xyz1234)-N.array(xyz))**2,1))))

    def saveTestData(self):
        path = QFileDialog.getSaveFileName(
                self, 'Save File', '', 'CSV(*.csv)')[0]
        if path:
            self.results.to_csv(path,sep=',',index=False)

    """
    Utility functions
    """
    def errorMsg(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(self.errMessage)
        msg.setWindowTitle(self.errorTitle)
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        retval = msg.exec_()        

app = QApplication([])
ex = MainWindow()
ex.show()
sys.exit(app.exec_())
