from PyQt5.QtWidgets import (QMainWindow, QApplication, QSlider,QColorDialog,
        QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsScene, QTableWidget,QTableWidgetItem, QVBoxLayout,
        QGraphicsItem, QGraphicsView, QVBoxLayout, QGridLayout,QGraphicsPixmapItem, QFrame,
        QPushButton,QTableView, QGraphicsItemGroup, QLabel, QFileDialog, QInputDialog, QLineEdit, QMessageBox, QGraphicsSimpleTextItem)
from PyQt5.QtGui import QPainter, QTransform, QColor, QPixmap, QStandardItemModel, QStandardItem, QIcon, QBrush
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot,QPointF, QPoint, QRectF

import sys
import csv
import os
import cv2
import numpy as N
import pandas as pd

import user_interface

class SelectMarkers:
    
    def __init__(self,imagePath,thresh,lowLim,upLim):
        
        self.point=1
        self.imagePath=imagePath
        self.threshold=thresh
        self.lowLim=lowLim
        self.upLim=upLim
        self.findContours()
        
    def findContours(self):

        inputFilepath = self.imagePath
        filename_w_ext = os.path.basename(inputFilepath)
        self.filename, file_extension = os.path.splitext(filename_w_ext)
        self.root=os.path.dirname(os.path.abspath(inputFilepath))
    
        self.image = cv2.imread(self.imagePath)
        imageGray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imageGray,self.threshold,255,cv2.THRESH_BINARY_INV)
    
        #Prevent a fatal crash because of version conflict in cv2.findContours
        try:
            
            self.cnts, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            ret, self.cnts, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)           
    
        self.good_cnts=[]
        for c in self.cnts:
            
            if self.lowLim <= cv2.contourArea(c) <= self.upLim:
                self.good_cnts.append(c)
                cv2.drawContours(self.image, [c], -5, (50, 600, 0), 1)
                # i=i+1

                (xstart, ystart, w, h) = cv2.boundingRect(c)
                M = cv2.moments(c)
                try:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    cv2.circle(thresh,(cx,cy),2,(255,255,255),-1)
                except ZeroDivisionError:
                    continue
            # self.point+=1

        self.path=self.root+"\\" + self.filename+'_contours.jpg'
        cv2.imwrite(self.path, thresh)

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

class Image(QGraphicsPixmapItem):
    """ create Pixmap image for each scene"""
    
    def __init__(self,path):
        super().__init__()
        
        self.setPixmap(QPixmap(path))

class Scene(QGraphicsScene):
    entered = pyqtSignal([QGraphicsItem],[QGraphicsEllipseItem])
    leave = pyqtSignal([QGraphicsItem],[QGraphicsEllipseItem])
    move = pyqtSignal([QGraphicsItem],[QGraphicsEllipseItem])
    """creates scenes to hold Pixmap image and circles """
    def __init__(self,path):
        super(Scene,self).__init__()
        self.path=path
        self.image = Image(self.path)        
        self.addItem(self.image)
        self.pointCount=0
        self.elp_x_pos=[]
        self.elp_y_pos=[]

    def remove_image(self):
        self.removeItem(self.image)
    
    def refresh_image(self,path):
        self.image = Image(self.path)        
        self.addItem(self.image)

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
        #This is relevant when there is a considerable perspective distortion.
        #Normalization: mean position at origin and mean distance equals to 1 at each direction.
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

        print('succeeded')
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
            raise ValueError('At least two sets of camera calibration parameters are needed for 3D point reconstruction.')

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
        self.tableCreated=False

        self.scenes=[]
        self.contour_scenes=[]
        self.calscenes=[]
        self.calibrations=[]
        self.viewPaths=[]
        self.calibrate=DLT(3,2)
        self.tableCreated=False

        self.initUI()

    def initUI(self):   
        
        #button connects
        self.addPointFile_b.clicked.connect(self.createTable)
        self.loadFilledTable_b.clicked.connect(self.loadTableData)
        # self.continue_b.clicked.connect(self.continueCal)
        self.newView_b.clicked.connect(self.add_scene)
        self.correct_b.clicked.connect(self.button_correct_centers)
        self.binary_b.clicked.connect(self.redo_binarization)
        self.saveTable_b.clicked.connect(self.saveTable)
        self.calibrate_b.clicked.connect(self.calibrate3D)
        self.loadTest_b.clicked.connect(self.loadTestTableData)
        self.testCal_b.clicked.connect(self.test3DCalibration)
        
        self.nmbViews1.valueChanged.connect(self.checkViews1)
        self.nmbViews2.valueChanged.connect(self.checkViews2)

        #list widget connects
        self.view_lw.clicked.connect(self.change_scene)
        self.contour_lw.clicked.connect(self.change_contour_scene)

        #other
        self.View.mouseDoubleClickEvent= self.add_point
        self.slider.setRange(1, 500)
        self.slider.setValue(100)
        self.slider.valueChanged[int].connect(self.onZoom)
        
        self.saveTest_b.clicked.connect(self.saveTestData)

    """
    User interface management methods
    """
    def checkViews1(self):

        if self.nmbViews1.value() > 1:
            self.addPointFile_b.setEnabled(True)
        if self.nmbViews1.value() < 2:
            self.addPointFile_b.setEnabled(False)
    
    def checkViews2(self):

        if self.nmbViews2.value() > 1:
            self.loadFilledTable_b.setEnabled(True)
        if self.nmbViews2.value() < 2:
            self.loadFilledTable_b.setEnabled(False)

    """
    Table loading and manipulation methods
    """
    def createTable(self):

        self.nmbViews=self.nmbViews1.value()

        pointFilePath=QFileDialog.getOpenFileName(self,"Load point data file",filter="Text files( *.txt *.csv)")

        if pointFilePath[0]!='':
            if self.tableCreated:
                self.tableWidget.clear() 
            self.num_points = sum(1 for line in open(pointFilePath[0]))
            self.tableWidget.setRowCount(self.num_points)                

            self.populateTableHeaders()
            
            #populate table with x,y,z data
            with open(pointFilePath[0]) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for count, row in enumerate(csv_reader):
                        self.tableWidget.setItem(count,0, QTableWidgetItem(row[0]))
                        self.tableWidget.setItem(count,1, QTableWidgetItem(row[1]))
                        self.tableWidget.setItem(count,2, QTableWidgetItem(row[2]))
            self.tableCreated=True
            self.groupBox_2.setEnabled(True) 
            self.groupBox_6.setEnabled(True)          

    def loadTableData(self):

        self.nmbViews=self.nmbViews2.value()

        pointFilePath=QFileDialog.getOpenFileName(self,"Load previously populated point data file",filter="Text files( *.txt *.csv)")

        if pointFilePath[0]!='':
            if self.tableCreated:
                self.tableWidget.clear()
            self.num_points = sum(1 for line in open(pointFilePath[0]))
            self.tableWidget.setRowCount(self.num_points)                
            self.populateTableHeaders()
 
            #populate table with x,y,z data
            try:
                with open(pointFilePath[0]) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    for row, rowData in enumerate(csv_reader):
                        for column in range(self.tableWidget.columnCount()):
                            self.tableWidget.setItem(row,column, QTableWidgetItem(rowData[column]))
                self.tableCreated=True
                self.groupBox_2.setEnabled(True)    
                self.groupBox_6.setEnabled(True)          

            except IndexError:
                self.tableWidget.clear()
                self.errMessage="Missing columns. Either choose less views, or maybe you are trying to load a calibration marker coordinate file (only xyz)?"
                self.errorTitle="Cannot load this file"
                self.errorMsg()
                return None                        

            
    def saveTable(self):

        path = QFileDialog.getSaveFileName(
                self, 'Save File', '', 'CSV(*.csv)')[0]
        
        if path:
            with open(path, 'w', newline='') as stream:
                writer = csv.writer(stream)
                for row in range(self.tableWidget.rowCount()):
                    rowdata = []
                    for column in range(self.tableWidget.columnCount()):
                        item = self.tableWidget.item(row, column)
                        
                        if item is not None:
                            rowdata.append(
                                str(item.text()))
                            print(item.text())

                    writer.writerow(rowdata)

    def loadTestTableData(self):

        pointFilePath=QFileDialog.getOpenFileName(self,"Load test data file",filter="Text files( *.txt *.csv)")

        if pointFilePath[0]!='':
 
            self.num_points = sum(1 for line in open(pointFilePath[0]))
            self.test_tableWidget.setRowCount(self.num_points)                
            self.populateTestTableHeaders()
 
            #populate table with x,y,z data
            with open(pointFilePath[0]) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row, rowData in enumerate(csv_reader):
                    for column in range(self.test_tableWidget.columnCount()):
                        self.test_tableWidget.setItem(row,column, QTableWidgetItem(rowData[column]))
    
            self.test_tableCreated=True    

    def populateTestTableHeaders(self):
        
        self.test_tableWidget.setColumnCount(3+2*self.nmbViews)

        headers=['x', 'y', 'z']
        
        for i in range(self.nmbViews):
            
            headers.append('v%s_x_px' % str(i+1))
            headers.append('v%s_y_px' % str(i+1))
        
        self.test_tableWidget.setHorizontalHeaderLabels(headers)
        style = "::section {""background-color: lightgrey; }"
        style = "::section {""background-color: darkgrey; }"
        self.test_tableWidget.horizontalHeader().setStyleSheet(style)
        self.test_tableWidget.verticalHeader().setStyleSheet(style)

    def populateTableHeaders(self):
        
        self.tableWidget.setColumnCount(3+2*self.nmbViews)

        headers=['x', 'y', 'z']
        
        for i in range(self.nmbViews):
            
            headers.append('v%s_x_px' % str(i+1))
            headers.append('v%s_y_px' % str(i+1))
        
        self.tableWidget.setHorizontalHeaderLabels(headers)
        style = "::section {""background-color: lightgrey; }"
        style = "::section {""background-color: darkgrey; }"
        self.tableWidget.horizontalHeader().setStyleSheet(style)
        self.tableWidget.verticalHeader().setStyleSheet(style)


    """
    View and Scene related methods
    """
    def add_scene(self):

        try:
            if len(self.scenes)>=int(self.nmbViews):
                self.errMessage="Cannot add anymore views. Please start again and specify the correct number of views."
                self.errorTitle="Cannot add anymore views."
                self.errorMsg()
                return None               
        except AttributeError:
            pass

        if self.view_lw.count() == 0:
            
            viewPath=QFileDialog.getOpenFileName(self,
            "Load image view",filter="Image Files( *.png *.jpg)")

            if viewPath[0]!='':
 
                self.current_scene=Scene(viewPath[0])
                self.current_scene.entered.connect(self.displayInfo)
                self.current_scene.leave.connect(self.removeInfo)
                self.current_scene.move.connect(self.ellipseMove)
                self.scenes.append(self.current_scene)

                self.View.setScene(self.current_scene)

                for count, scene in enumerate(self.scenes):
                    if self.current_scene==self.scenes[count]:
                        self.current_scene_idx=count
                base=os.path.basename(viewPath[0])
                self.view_lw.addItem("%s_V%s" %(os.path.splitext(base)[0],str(self.current_scene_idx+1)))

                self.sceneCount=len(self.scenes)
                self.view_lw.item(self.sceneCount-1).setSelected(True)
                self.view_lw.setCurrentRow(self.sceneCount-1)
                self.newView_b.setText("Add another view")
                self.viewPaths.append(viewPath)
                self.correct_centers(way_in="internal")
                self.contour_lw.setCurrentRow(self.sceneCount-1)

        else:

            viewPath=QFileDialog.getOpenFileName(self,
                        "Load image view",filter="Image Files( *.png *.jpg)")

            if viewPath[0]!='':
    
                self.current_scene=Scene(viewPath[0])
                self.current_scene.entered.connect(self.displayInfo)
                self.current_scene.leave.connect(self.removeInfo)
                self.current_scene.move.connect(self.ellipseMove)
                self.scenes.append(self.current_scene)

                for count, scene in enumerate(self.scenes):
                    if self.current_scene==self.scenes[count]:
                        self.current_scene_idx=count

                base=os.path.basename(viewPath[0])
                self.view_lw.addItem("%s_V%s" %(os.path.splitext(base)[0],str(self.current_scene_idx+1)))
                self.View.setScene(self.current_scene)

                self.sceneCount=len(self.scenes)
                self.view_lw.item(self.sceneCount-1).setSelected(True)
                self.view_lw.setCurrentRow(self.sceneCount-1)
                self.viewPaths.append(viewPath)
                self.correct_centers(way_in="internal")
                self.contour_lw.setCurrentRow(self.sceneCount-1)

    def change_scene(self):
        self.current_scene=self.scenes[self.view_lw.currentRow()]
        self.View.setScene(self.current_scene)
        self.current_scene_idx=self.view_lw.currentRow()

    def change_contour_scene(self):
        self.current_scene=self.contour_scenes[self.contour_lw.currentRow()]
        self.View.setScene(self.current_scene)
        # self.current_scene_idx=self.view_lw.currentRow()

    def redo_binarization(self):

        
        self.cal_current_scene=[]
        self.lowLim=int(self.lowContour_le.text())
        self.upLim=int(self.upContour_le.text())
        self.thresh=int(self.binary_le.text())

        image=SelectMarkers(self.viewPaths[self.contour_lw.currentRow()][0],self.thresh,self.lowLim,self.upLim)

        self.calibrations[self.contour_lw.currentRow()]=image
        self.cal_current_scene=Scene(image.path)
        contourScene='%s_ctrs' % (self.view_lw.item(self.contour_lw.currentRow()).text())               
        i=self.contour_lw.findItems(contourScene, Qt.MatchExactly)
    
        self.contour_scenes[self.contour_lw.indexFromItem(i[0]).row()]=self.cal_current_scene
    
        self.change_contour_scene()

    def onZoom(self, value):
    
        val = value / 100    
        self.View.resetTransform()
        self.View.scale(val, val)    


    """
    Calibration marker methods
    """
    def add_point(self, event):
        
        if self.current_scene.pointCount == self.num_points:
            self.errMessage="You have placed all the points listed in the calibration point file. If you wish to add more points, add more rows to the point file and start over."
            self.errorTitle="Cannot place any more points"
            self.errorMsg()
            return None

        if self.current_scene_idx > self.nmbViews-1:
            return None
        
        p=self.View.mapToScene(event.x(),event.y())
        self.current_scene.pointCount=self.current_scene.pointCount+1
        self.circle=Ellipse(0,0, 10, 10,self.current_scene.pointCount)
        self.circle.setPos(p.x()-5,p.y()-5)
        self.current_scene.addItem(self.circle)
        
        x=str(p.x()-5)
        y=str(p.y()-5)            
        self.tableWidget.setItem(self.current_scene.pointCount-1,self.current_scene_idx+3+self.current_scene_idx*1, QTableWidgetItem(x))
        self.tableWidget.setItem(self.current_scene.pointCount-1,self.current_scene_idx+4+self.current_scene_idx*1, QTableWidgetItem(y))

    def button_correct_centers(self):

        self.correct_centers(way_in='button')

    def correct_centers(self, way_in):
        
        self.cal_current_scene=[]
        self.lowLim=int(self.lowContour_le.text())
        self.upLim=int(self.upContour_le.text())
        self.thresh=int(self.binary_le.text())
        

        if way_in == "internal":
            try:

                image=SelectMarkers(self.viewPaths[self.view_lw.currentRow()][0],self.thresh,self.lowLim,self.upLim)
                self.calibrations.append(image)
                self.cal_current_scene=Scene(image.path)
                contourScene='%s_ctrs' % (self.view_lw.item(self.view_lw.currentRow()).text())

                self.contour_scenes.append(self.cal_current_scene)
                self.contour_lw.addItem(contourScene)

            except AttributeError:
                self.errMessage="Please select an original view (not binarized) to manipulate"
                self.errorTitle="Select an original view"
                self.errorMsg()
                return None

        elif way_in=="button":

            try:

                image=SelectMarkers(self.viewPaths[self.contour_lw.currentRow()][0],self.thresh,self.lowLim,self.upLim)

                self.calibrations[self.contour_lw.currentRow()]=image
                self.cal_current_scene=Scene(image.path)
                contourScene='%s_ctrs' % (self.view_lw.item(self.contour_lw.currentRow()).text())               
                i=self.contour_lw.findItems(contourScene, Qt.MatchExactly)
            
                self.contour_scenes[self.contour_lw.indexFromItem(i[0]).row()]=self.cal_current_scene

            except AttributeError:
                self.errMessage="Please select an original view (not binarized) to manipulate"
                self.errorTitle="Select an original view"
                self.errorMsg()
                return None            

            
        for count, scene in enumerate(self.scenes):
            if self.current_scene==self.scenes[count]:
                self.current_scene_idx=count

        self.sceneCount=len(self.scenes)

        #add ellipse positions to scenes

        for idx, i in enumerate(self.scenes[self.view_lw.currentRow()].items()):
            
            if isinstance(i, Ellipse):
                
                for c in self.calibrations[self.view_lw.currentRow()].good_cnts:
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
                            self.current_scene_idx=self.view_lw.currentRow()
                            self.ellipseMove(i)
                            i.setBrush(QColor('green'))
                            
                        except ZeroDivisionError:
                            continue

        self.current_scene_idx=self.view_lw.currentRow()
        self.change_scene()
 
    def displayInfo(self, item):
        
        self.text=QGraphicsSimpleTextItem("P_%s" % str(item.count))
        self.text.setBrush(Qt.blue)
        self.text.setPos(item.pos().toPoint().x()+10, item.pos().toPoint().y()+10)
        self.current_scene.addItem(self.text)

    def removeInfo(self, item):
        
        self.current_scene.removeItem(self.text)

    def ellipseMove(self, item):

        item.setBrush(QColor('red'))
        self.current_scene.removeItem(self.text)
        x=str(item.pos().toPoint().x()+5)
        y=str(item.pos().toPoint().y()+5)           
        self.tableWidget.setItem(item.count-1,self.current_scene_idx+3+self.current_scene_idx*1, QTableWidgetItem(x))
        self.tableWidget.setItem(item.count-1,self.current_scene_idx+4+self.current_scene_idx*1, QTableWidgetItem(y))


    """
    Calibration and testing methods
    """
    def calibrate3D(self):

        #set up xyz coords list
        xyz = []
        for row in range(self.tableWidget.rowCount()):
            container=[]
            for column in range(3):
                item = self.tableWidget.item(row, column)
                
                container.append(float(item.text()))
            xyz.append(container)
        
        #list of px coord lists
        px_coords = []
        for view in range(self.nmbViews):
            view_coords=[]
            for row in range(self.tableWidget.rowCount()):
                
                x_column = 3 + 2*(view)
                y_column = 4 + 2*(view)
                x_item = self.tableWidget.item(row, x_column)
                y_item = self.tableWidget.item(row, y_column)
                coords=[float(x_item.text()),float(y_item.text())]
                view_coords.append(coords)

            px_coords.append(view_coords)
        
        self.nmbDim=3
        self.coefficients=[]
        errors=[]
        
        self.nmbCam=self.nmbViews
        

        #get parameters for each view
        for view in px_coords:
            L, err = self.calibrate.DLTcalib(self.nmbDim, xyz, view)
            self.coefficients.append(L)
            errors.append(err)
            
    def test3DCalibration(self):            

        #set up xyz coords list
        xyz = []
        for row in range(self.test_tableWidget.rowCount()):
            container=[]
            for column in range(3):
                item = self.test_tableWidget.item(row, column)
                
                container.append(float(item.text()))
            xyz.append(container)
        
        #list of px coord lists
        px_coords = []
        for view in range(self.nmbViews):
            view_coords=[]
            for row in range(self.test_tableWidget.rowCount()):
                
                x_column = 3 + 2*(view)
                y_column = 4 + 2*(view)
                x_item = self.test_tableWidget.item(row, x_column)
                y_item = self.test_tableWidget.item(row, y_column)
                coords=[float(x_item.text()),float(y_item.text())]
                view_coords.append(coords)

            px_coords.append(view_coords)

        xyz1234 = N.zeros((len(xyz),3))
        #use parameters to reconstruct input points 
        for i in range(len(px_coords[0])):
            i_px_coords=[]
            for view in px_coords:
                i_px_coords.append(view[i])

            xyz1234[i,:] = self.calibrate.DLTrecon(self.nmbDim,self.nmbCam,self.coefficients,i_px_coords)

        rec_xyz=pd.DataFrame(xyz1234,columns=['rec_x','rec_y','rec_z'])
        xyz=pd.DataFrame(xyz,columns=['x','y','z'])
        self.results=pd.concat([xyz,rec_xyz], axis=1)
        self.results['x_e']=self.results['x']-self.results['rec_x']
        self.results['y_e']=self.results['y']-self.results['rec_y']
        self.results['z_e']=self.results['z']-self.results['rec_z']

        self.xmean_l.setText(str(((self.results.x - self.results.rec_x)**2).mean()**.5))
        self.ymean_l.setText(str(((self.results.y - self.results.rec_y)**2).mean()**.5))
        self.zmean_l.setText(str(((self.results.z - self.results.rec_z)**2).mean()**.5))

        self.pointsTested_l.setText(str(self.test_tableWidget.rowCount()))
        # self.ystd_l.setText(str(self.results['y_e'].std()))
        # self.zstd_l.setText(str(self.results['z_e'].std()))



        # print(x_std,y_std,z_std)
        print(self.results)

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

    def continueCal(self):
        
        if self.nmbViews < 2:
            self.errMessage="Please specify at least 2 views"
            self.errorTitle="Not enough views"
            self.groupBox_2.setDisabled(True)
            self.addPointFile_b.setEnabled(True)
            self.nmbViews.setEnabled(True)
            self.errorMsg()
            return None  
        
        if not self.tableCreated:
            self.errMessage="Please choose calibration point file before continuing"
            self.errorTitle="Calibration file not specified"
            self.groupBox_2.setDisabled(True)
            self.addPointFile_b.setEnabled(True)
            self.nmbViews.setEnabled(True)
            self.errorMsg()
            return None   
        else:
            self.continue_b.setDisabled(True)    

           
app = QApplication([])
ex = MainWindow()
ex.show()
sys.exit(app.exec_())


