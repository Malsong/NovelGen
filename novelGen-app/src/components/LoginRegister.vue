<template>
<div class="lr-container">
  <div class="login-register" >
        <div class="con-box left" >
            <h2>欢迎来到<span>创作小镇</span></h2>
            <p>快来创建你的专属<span>世界</span>吧</p>
            <img src="../assets/8.png" alt="">
            <p>已有账号</p>
            <button id="login" @click="changeType" >去登录</button>
        </div>
        <div class="con-box right">
            <h2>欢迎来到<span>创作小镇</span></h2>
            <p>快来继续创建你的专属<span>世界</span>吧</p>
            <img src="../assets/01.png" alt="">
            <p>没有账号？</p>
            <button id="register" @click="changeType" >去注册</button>
        </div>

        <!-- form-box为需移动的盒子 -->
        <div class="form-box" :class="{ active0: isLogin, active1: !isLogin }">
            <!-- 登录 -->
            <div class="login-box" v-if="isLogin">
                <h1 class="h1_lr">login</h1>
                <input class="inp_lr" type="text" placeholder="用户名" v-model="form.username">
                <input class="inp_lr" type="password" placeholder="密码" v-model="form.userpwd">
                <button @click="login">登录</button>
            </div>
            <!-- 注册 -->
            <div class="register-box" v-else>
                <h1 class="h1_lr">register</h1>
                <input class="inp_lr" type="text" placeholder="用户名" v-model="form.username">
                <input class="inp_lr" type="password" placeholder="密码" v-model="form.userpwd">
                <input class="inp_lr" type="number" placeholder="年龄" min="1" v-model="form.userage">
                <input class="inp_lr tgl tgl-skewed" id='cb3' type='checkbox' v-model="form.usersex">
		        <label class='tgl-btn' data-tg-off='男' data-tg-on='女' for='cb3'></label>
                <button @click="register">注册</button>
            </div>
        </div>
    </div>
    </div>
</template>

<script>
/* import qs from "qs" */
export default {
    name:"login-register",
    data(){
        return{
            isLogin:true,
            userError:false,
            pwdError:false,
            existed: false,
            form:{
                username:"",
                userpwd:"",
                userage:18,
                usersex:false
            }
        }
    },
    methods:{
        changeType() {
            this.isLogin = !this.isLogin,
            this.form.username = '',
            this.form.userpwd = ''
		},
        login(){
            const self = this;
            if (self.form.username !="" && self.form.userpwd != ""){
                let params=self.$qs.stringify({name:self.form.username,passwd:self.form.userpwd})
                self.$http({
                    method:'post',
                    url:'http://localhost:5000/api/login',
                    data:params
                })
                .then(res =>{
                    /* console.log('res=>',res);
                    console.log('res=>',res.data); */
                    if(typeof(res.data)=="object")
                    {
                        this.$store.dispatch('login',res.data).then(()=>{
                            this.$notify({
                                type:'success',
                                message:'欢迎你，'+res.data['username']+'!',
                                duration:3000
                            })
                            this.$router.replace('/generate')
                        })
                    }
                    else{
                        switch(res.data){
                            case 0:
                                this.$notify({
                                    type:'error',
                                    message:'系统繁忙，请稍后再试！',
                                    duration:3000
                                })
                                /* alert("系统繁忙，请稍后再试！"); */
                                break;
                            case 2:
                                this.userError = true;
                                this.$notify({
                                    type:'error',
                                    message:'无此用户“'+self.form.username+'”，请注册！',
                                    duration:3000
                                })
                                /* alert("无此用户“"+self.form.username+"”，请注册！"); */
                                break;
                            case 3:
                                this.pwdError = true;
                                this.$notify({
                                    type:'error',
                                    message:'用户“'+self.form.username+'”，密码错误！',
                                    duration:3000
                                })
                                /* alert("用户“"+self.form.username+"”密码错误，请重新输入!"); */
                                break;
                        }
                    }
                    
                } )
                .catch( err =>{
                    console.log(err);
                })
            }
            else{
                this.$notify({
                    type:'error',
                    message:'填写不能为空！',
                    duration:3000
                })
            }
        },
        register(){
            const self = this;
            if(self.form.username !="" && self.form.userpwd != "" && self.form.userage != null && self.form.usersex != null){
                let params=self.$qs.stringify({name:self.form.username,passwd:self.form.userpwd,age:self.form.userage,sex:self.form.usersex})
                self.$http({
                    method:'post',
                    url:'http://localhost:5000/api/register',
                    data:params
                })
                .then(res =>{
                    switch(res.data){
                        case 0:
                            this.$notify({
                                type:'error',
                                message:'系统繁忙，请稍后再试！',
                                duration:3000
                            })
                            /* alert("系统繁忙，请稍后再试！"); */
                            break;
                        case 1:
                            this.$notify({
                                type:'error',
                                message:'用户“'+self.form.username+'”注册成功！',
                                duration:3000
                            })
                            /* alert("用户“"+self.form.username+"”注册成功!"); */
                            break;
                        case 2:
                            this.existed = true;
                            this.$notify({
                                type:'error',
                                message:'用户已存在“'+self.form.username+'”，请重新注册！',
                                duration:3000
                            })
                            /* alert("用户已存在“"+self.form.username+"”，请重新注册！"); */
                            break;
                    }
                })
                .catch( err =>{
                    console.log(err);
                })
            }
            else{
                this.$notify({
                    type:'error',
                    message:'填写不能为空！',
                    duration:3000
                })
            }
        }
    }
}
</script>

<style>
.lr-container{
    /* 100%窗口高度 */
    height: 100vh;
    /* 弹性布局 水平+垂直居中 */
    display: flex;
    justify-content: center;
    align-items: center; 
}
.login-register{
    background-color: #fff;
    width: 650px;
    height: 415px;
    border-radius: 5px;
    /* 阴影 */
    box-shadow: 5px 5px 5px rgba(0,0,0,0.1);
    /* 相对定位 */
    position: relative;
}
.form-box{
    /* 绝对定位 */
    position: absolute;
    top: -10%;
    left: 5%;
    background-color: #d1c8fd;/*#c2b8f4;*/
    width: 320px;
    height: 500px;
    border-radius: 5px;
    box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 2;
    /* 动画过渡 加速后减速 */
    transition: 0.5s ease-in-out;
}
.register-box,.login-box{
    /* 弹性布局 垂直排列 */
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
}
.sexbox{
    width: 100%;
    display: flex;
    /* align-items: center; */
}
.h1_lr{
    text-align: center;
    margin-bottom: 25px;
    /* 大写 */
    text-transform: uppercase;
    color: #fff;
    /* 字间距 */
    letter-spacing: 5px;
}
.inp_lr{
    background-color: transparent;
    width: 70%;
    color: #fff;
    border: none;
    /* 下边框样式 */
    border-bottom: 1px solid rgba(255,255,255,0.4);
    padding: 10px 0;
    text-indent: 10px;
    margin: 8px 0;
    font-size: 14px;
    letter-spacing: 2px;
}
.inp_lr::placeholder{
    color: #fff;
}
.inp_lr:focus{
    color: #a899EF;
    outline: none;
    border-bottom: 1px solid #a899EF;
    transition: 0.5s;
}
.inp_lr:focus::placeholder{
    opacity: 0;
}
.form-box button{
    width: 70%;
    margin-top: 35px;
    background-color: #f6f6f6;
    outline: none;
    border-radius: 8px;
    padding: 13px;
    color: #a899EF;
    letter-spacing: 2px;
    border: none;
    cursor: pointer;
}
.form-box button:hover{
    background-color: #a899EF;
    color: #f6f6f6;
    transition: background-color 0.5s ease;
}
.form-box.active0{
    transform: translateX(0%);
}
.form-box.active1{
    transform: translateX(80%);
}
.con-box{
    width: 50%;
    /* 弹性布局 垂直排列 居中 */
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    /* 绝对定位 居中 */
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
}
.con-box.left{
    left: -2%;
}
.con-box.right{
    right: -2%;
}
.con-box h2{
    color: #8e9aaf; /*大部分字的颜色 */
    font-size: 25px;
    font-weight: bold;
    letter-spacing: 3px;
    text-align: center;
    margin-bottom: 4px;
}
.con-box p{
    font-size: 12px;
    letter-spacing: 2px;
    color: #8e9aaf;
    text-align: center;
}
.con-box span{
    color: #d1c8fd; /*重点字颜色 */
}
.con-box img{
    width: 150px;
    height: 150px;
    opacity: 0.9;
    margin: 40px 0;
}
.con-box button{
    margin-top: 3%;
    background-color: #fff;
    color: #a899EF;
    border: 1px solid #d1c8fd;
    padding: 6px 10px;
    border-radius: 5px;
    letter-spacing: 1px;
    outline: none;
    cursor: pointer;
}
.con-box button:hover{
    background-color: #d1c8fd;
    color: #fff;
}

.tgl { display: none; }
.tgl + .tgl-btn { 
    margin-top: 3%;
    outline: 0; 
    display: block; 
    width:60%; 
    height:28px; 
    border-radius: 5px;
    position: relative; 
}
.tgl-skewed + .tgl-btn { 
    overflow: hidden;
        font-family: sans-serif; 
}
.tgl-skewed + .tgl-btn:after, .tgl-skewed + .tgl-btn:before { 
    display: inline-block; 
        width: 50%; 
        text-align: center; 
        position: absolute; 
        line-height: 30px; 
        font-weight: bold; 
        color: #fff; 
        font-size: 14px;
}
/*默认的*/
.tgl-skewed + .tgl-btn:after { 
    left: 0;
    color: white; /* #40404C; */
    background-color: #a899EF;
    content: attr(data-tg-on); 
}	
.tgl-skewed + .tgl-btn:before { 
    color:  #AFB9CA;
    left: 50%; 
    background-color: #F7F9FC;
    content: attr(data-tg-off); 
}
/*选中后*/
.tgl-skewed:checked + .tgl-btn:after { 
    left: 50%;
    background-color: #a899EF;  
    content: attr(data-tg-off); 
}	
.tgl-skewed:checked + .tgl-btn:before { 
    left: 0;
    background-color: #F7F9FC;
    content: attr(data-tg-on);
}

.inp_lr::-webkit-outer-spin-button,
    .inp_lr::-webkit-inner-spin-button {
        -webkit-appearance: none;
    }
    .inp_lr[type="number"]{
        -moz-appearance: textfield;
    }
</style>