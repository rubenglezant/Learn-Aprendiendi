# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 02:29:57 2019

@author: J. Albero Gonzalez P
"""
import time

def openHelper():
    f = open("../output/report.html","w")
    f.write("<html>\n")
    f.write("<body>\n")
    return f

def appendHelper():
    f = open("../output/report.html","a")
    return f

def close_withHTMLopen(f):
    f.close()

def closeHelper(f):
    f.write("</body>\n")
    f.write("</html>\n")
    f.close()

def main_title(f,s):
    f.write("<h1>\n")
    f.write(s+"\n")
    f.write("</h1>\n")

def title_3(f,s):
    f.write("<h3>\n")
    f.write(s+"\n")
    f.write("</h3>\n")

def title_2(f,s):
    f.write("<h2>\n")
    f.write(s+"\n")
    f.write("</h2>\n")

def text(f,s):
    f.write("<p>\n")
    f.write(s+"\n")
    f.write("</p>\n")

def imagen(f,img):
    current_milli_time = lambda: int(round(time.time() * 1000))
    tAhora = current_milli_time()
    name = "fig-"+str(tAhora)+".png"
    direc = '../output/'
    img.savefig(direc + name)
    f.write("<img src='"+name+"'/>")

def title_text(f,s,s2):
    f.write("<h3>\n")
    f.write(s+"\n")
    f.write("</h3>\n")
    f.write("<p>\n")
    f.write(s2+"\n")
    f.write("</p>\n")
