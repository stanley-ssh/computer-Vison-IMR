# computer-Vison-IMR ( using Open-CV)



## The goal post detection( for both the blue and yellow goal post ) 

  -The goal detection was done by segmenting the color of the goalpost which
  created the mask. After which i found the contours of each image. The area
  of the goal was calculated by the getting the area of the contour value. To
  calculate the points (X, Y), We calculated the arclength of the contour values.
   These points were then used to draw a bounding rectangle on the goalpost.
   
 ## The Obstacle Detection:
 
 -  The Obstacle detection was similar to the goal post detection, with the difference being
that the obstacles were of two colors, one of the colors like that of the
background of the image (white). I extracted the red boxes on the obstacle and then extracted some parts of the white boxes.
This white box extraction wasn’t very clean, 
so I used the opening morphological operation to clean it up and added the red mask to the white mask after cleaning up.
After which I used this new added mask to get the contour of the obstacle.


##Line Segement Detection

-For the line segement, i segmented the green field and then got the white lines on
the field. The lines where detected using the OpenCV fast line detector. This
function gave the list of lines traced by the white line on the field. To get
a single line I will need to merge the parallel line and the intersecting lines.
This hasnt been done yet for this question.


Note: Line classification has not been fully implemented
