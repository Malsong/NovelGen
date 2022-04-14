from asyncio.windows_events import NULL
from turtle import title
""" from crypt import methods """
from sqlite3 import connect
from statistics import mode
from unicodedata import name
from flask import Flask, request, jsonify, make_response
from flask_cors import *

""" import time """
import pymysql
""" import json
import copy

import os
import subprocess
import shutil """

from py_moduels.generateNovel import generationNovel

def conn_mysql(sql):
    conn = pymysql.connect("localhost","root","160190651sw?!","novelgeninfo",charset='utf8')
    cursor = conn.cursor()
    cursor.execute(sql)
    res = cursor.fetchall()
    """ print(res) """
    conn.commit()
    cursor.close()
    conn.close()
    return res

def conn_change(sql):
    conn = pymysql.connect("localhost","root","160190651sw?!","novelgeninfo",charset='utf8')
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.commit()
    cursor.close()
    conn.close()

app = Flask(__name__,static_url_path='/static/',template_folder='templates')
CORS(app, resources=r'/*')
app.debug = True

@app.route('/api/login', methods=['POST'])
def login():
    '''
    登录验证
    '''
    if request.method == 'POST':
        name = request.form['name']
        passwd = request.form['passwd']
    try:
        sql = '''select name,passwd,age,sex from users where name="{}"'''.format(name)
        rs = conn_mysql(sql)
    except:
        rs = 'db-error'

    """0:数据库繁忙 1:表示识别成功 2:表示无此用户 3:表示密码错误"""
    if rs == 'db-error':
        return '0'  
    elif rs == ():
        return '2'
    elif rs[0][1]==passwd:
        userData={}
        userData['username']=rs[0][0]
        userData['passwd']=rs[0][1]
        userData['age']=rs[0][2]
        userData['sex']=rs[0][3]
        return make_response(jsonify(userData))
    else:
        return '3'

@app.route('/api/register', methods=['POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        passwd = request.form['passwd']
        age = request.form['age']
        sex = request.form['sex']
    db=pymysql.connect("localhost","root","160190651sw?!","novelgeninfo",charset='utf8')
    cursor=db.cursor()
    id_max = 0
    try:
        sql = '''select * from users where name="{}"'''.format(name)
        cursor.execute(sql)
        rs = cursor.fetchall()
    except:
        rs = 'db-error'
        print('py-db-error')
    # 0:繁忙 1:成功 2:存在
    if rs==():
        try:
            sql_max = '''select MAX(id) from users'''
            cursor.execute(sql_max)
            id_max = cursor.fetchall()[0][0]
            print(id_max)
            if sex == True:
                sexnum = 1
            else:
                sexnum = 0
            sql_in='''INSERT INTO users VALUES ({},"{}","{}",{},{});'''.format(id_max+1,name,passwd,age,sexnum)
            print(sql_in)
            if cursor.execute(sql_in):
                db.commit()
                print("数据插入成功！")
            db.close()
            return '1'
        except:
            rs = 'db-error'

    db.close()
    if rs != 'db-error':
        return '2'
    else:
        return '0'

@app.route('/api/generateText', methods=['POST'])
def generateText():
    if request.method == 'POST':
        model_class = request.form['m_class']
        prefix = request.form['prefix']
        length =  request.form['length']
        temperature = request.form['temperature']
        prelen = len(request.form['prefix'])
    """ prefix = prefix.replace('\n','[SEP]') """
    count = prefix.count('\n')  #换行符个数
    print(model_class)
    print(prefix)
    print(length)
    """ print(temperature) """
    dict_model = { 'A':'./public/models/A10', 'B':'./public/models/B15', 'C':'./public/models/C20', 'E':'./public/models/E15', 'F':'./public/models/F15', 'G':'./public/models/G15'}
    if model_class != '':
        model_path = dict_model[model_class]
    textList = generationNovel(prefix,model_path,length,temperature)
    textDict ={}
    for i in range(len(textList)):
        textDict[i]=textList[i][prelen-count:]
    print(textDict)
    return make_response(jsonify(textDict))

@app.route('/api/saveText',methods=['POST'])
def saveText():
    if request.method == 'POST':
        name = request.form['name']
        date = request.form['date']
        title = request.form['title']
        text =  request.form['text']
    try:
        sql_in='''INSERT INTO novel VALUES ("{}","{}","{}","{}");'''.format(date,name,title,text)
        conn_change(sql_in)
        print("数据插入成功！")
        return '1'
    except:
        print("数据插入失败！")
        return '0'

@app.route('/api/getText',methods=['POST'])
def getText():
    if request.method == 'POST':
        username = request.form['name']
    try:
        sql = '''select * from novel where name="{}"'''.format(username)
        rs = conn_mysql(sql)
    except:
        rs = 'db-error'
        print('py-db-error')
    tableData = []
    try:
        for item in rs:
            oneData={}
            oneData['date']=item[0]
            oneData['txtTitle']=item[2]
            oneData['txtText']=item[3]
            tableData.append(oneData)
    except:
        pass
    """ print(rs)
    print(tableData) """
    return make_response(jsonify(tableData))

@app.route('/api/deleteText',methods=['POST'])
def deleteText():
    if request.method == 'POST':
        username = request.form['name']
        textTitle = request.form['title']
    try:
        sql = '''delete from novel where name="{}" and title="{}"'''.format(username,textTitle)
        conn_change(sql)
        print("数据删除成功！")
        return '1'
    except:
        print("数据删除失败！")
        return '0'

@app.route('/api/haveSame',methods=['POST'])
def haveSame():
    if request.method == 'POST':
        name = request.form['name']
        title = request.form['title']
    try:
        sql = '''select * from novel where name="{}" and title="{}"'''.format(name,title)
        rs = conn_mysql(sql)
        if rs!=():
            return '1'
        else: return '2'
    except:
        return '0'
    

@app.route('/api/coverText',methods=['POST'])
def coverText():
    if request.method == 'POST':
        name = request.form['name']
        date = request.form['date']
        title = request.form['title']
        text =  request.form['text']
    try:
        sql='''update novel set date = "{}",text="{}" where name="{}" and title="{}"'''.format(date,text,name,title)
        conn_change(sql)
        print("数据覆盖成功！")
        return '1'
    except:
        print("数据覆盖失败！")
        return '0'

@app.route('/api/changePwd',methods=['POST'])
def changePwd():
    if request.method == 'POST':
        username = request.form['name']
        passwd = request.form['newpwd']
    try:
        sql='''update users set passwd = "{}" where name="{}"'''.format(passwd,username)
        conn_change(sql)
        print("密码修改成功！")
        return '1'
    except:
        print("密码修改失败！")
        return '0'

# 主页
@app.route('/')
def index():
    return "Hi"

if __name__ == '__main__':
  app.run(debug = True)