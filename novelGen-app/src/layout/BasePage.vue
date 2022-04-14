<template>

<!-- @select="handleSelect" -->
    <div class="navbar">
    <el-menu
    :default-active="$route.path"
    mode="horizontal"
    class="HeadMenu"
    router
    active-text-color="#a899ef"
    >
        <el-menu-item index="backtohome">
          <el-icon><home-filled /></el-icon>
            <span @click="checkToHome">首页</span>
        </el-menu-item>
        <el-menu-item index="/generate">
          <el-icon><edit/></el-icon>
          <span>创作</span>
        </el-menu-item>
        
        <el-menu-item index="/save" v-show="$store.state.stateOn">
          <el-icon><folder-opened /></el-icon>
          <span>作品集</span>
        </el-menu-item>
   
        <el-sub-menu index='other' v-show="$store.state.stateOn">
          <template #title><el-icon><avatar /></el-icon><span>个人</span></template>
          <!-- <el-menu-item index="/userP"> -->
          <el-menu-item index="getuserInfo">
            <el-icon><user/></el-icon>
            <span @click="getInfo">个人信息</span>
          </el-menu-item>

          <el-menu-item index="changeInfo">
            <el-icon><list /></el-icon>
            <span @click="changepwd">修改密码</span>
          </el-menu-item>

          <el-menu-item index="userlogout">
            <el-icon><switch-button /></el-icon>
              <!-- <el-button @click.native="logOut">退出登录</el-button> -->
              <span @click="logOut">退出登录</span>
          </el-menu-item>
        </el-sub-menu>
        
        <el-menu-item index="/login" v-show="!$store.state.stateOn">
          <el-icon><switch-button /></el-icon>
          <span>登录</span>
        </el-menu-item>
        
    </el-menu>
    </div>
    
    <el-dialog v-model="dialogFormVisible" title="个人信息" :modal-append-to-body="false" width="300px" center>
      <el-form :align='center'>
        <el-form-item label="账户名:">
          <el-container v-if="$store.state.stateOn">{{$store.state.user.username}}</el-container>
        </el-form-item>
        <el-form-item label="年龄:">
         <el-container v-if="$store.state.stateOn">{{$store.state.user.age}}</el-container>
        </el-form-item>
        <el-form-item label="性别:">
         <el-container v-if="$store.state.stateOn">{{$store.state.user.sex==1?"男":"女"}}</el-container>
        </el-form-item>
      </el-form>
    </el-dialog>

    <el-dialog v-model="isChange" title="修改密码" width="300px" @close='closeDialog'>
    <el-form :model="pwd">
      <el-form-item label="原密码">
        <el-input v-model="pwd.oldpwd" autocomplete="off" type="password" show-password></el-input>
      </el-form-item>
      <el-form-item label="新密码">
        <el-input v-model="pwd.newpwd" autocomplete="off" type="password" show-password></el-input>
      </el-form-item>
    </el-form>
    <template #footer>
      <span class="dialog-footer">
        <el-button @click="closeDialog">取消</el-button>
        <el-button type="primary" @click="checkAndChange">确定</el-button>
      </span>
    </template>
    </el-dialog>
    <router-view></router-view>
   
</template>

<script>

/* import {ref} from 'vue'
const activeIndex = ref("/generate") 
//activeIndex
*/
export default {
  data(){
    return{
      dialogFormVisible:false,
      isChange:false,
      pwd:{
        oldpwd:'',
        newpwd:''
      }
      
    }
  },
  methods:{
    checkToHome(){
      if(this.$store.state.stateOn==false){
        this.$store.dispatch('logOut').then(()=>{this.$router.replace('/')})
      }
      else{
        this.$confirm('此操作将退出登录，确定退出?', '提示', {
            confirmButtonText: '确定',
            cancelButtonText: '取消',
            type: 'warning'
          }).then(() => {
            this.$message({
              type: 'success',
              message: '已退出登录!'
            });
            this.$store.dispatch('logOut').then(()=>{this.$router.replace('/')})
          }).catch(() => {
            this.$message({
              type: 'info',
              message: '已取消!'
            });
          })
        }
    },
    logOut(){ 
      this.$store.dispatch('logOut').then(()=>{
        this.$router.replace('/');
        this.$message({
              type: 'success',
              message: '已退出登录!'
            });
        }
      )
    },
    getInfo(){
      this.dialogFormVisible=true;
    },
    changepwd(){
      this.isChange=true;
    },
    closeDialog(){
      this.isChange=false
      this.pwd.oldpwd=''
      this.pwd.newpwd=''
    },
    checkAndChange(){
      if(this.pwd.oldpwd!='' & this.pwd.newpwd!='' ){
        if(this.pwd.oldpwd==this.$store.state.user.passwd){
          /* alert('可以修改') */
          //修改密码
          const self = this;
          let params=self.$qs.stringify({name:self.$store.state.user.username,newpwd:self.pwd.newpwd})
          self.$http({
            method:'post',
            url:'http://localhost:5000/api/changePwd',
            data:params
          })
          .then(res =>{
            switch(res.data){
              case 0:
                this.$notify({
                  type:'error',
                  message:'系统繁忙，请稍后再试!',
                  duration:3000
                })
                break;
              case 1:
                this.$notify({
                  type:'success',
                  message:'密码修改成功，请重新登录!',
                  duration:3000
                })
                this.$store.dispatch('logOut').then(()=>{this.$router.replace('/')})
                break;
            }
          } )
          .catch( err =>{
              console.log(err);
          })
        } 
        else{
          this.$notify({
            type:'error',
            message:'原密码验证错误!',
            duration:3000
          })
          this.pwd.oldpwd=''
          this.pwd.newpwd=''
        }
      }else{
        this.$notify({
          type:'error',
          message:'请填写完整!',
          duration:3000
        })
        this.pwd.oldpwd=''
        this.pwd.newpwd=''
      }
      
    }
  },
}
</script>

<style>

</style>