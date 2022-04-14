<template>
 <div style="height:100vh;">
  <el-container class="gobuttonBox">
    <el-button @click="goto1" size="large" color="#a899ef" plain>用户入口</el-button>
    <el-button @click="goto2" size="large" color="#a899ef" plain>游客入口</el-button>
  </el-container>
  </div>
</template>

<script>
export default {
  methods:{
    goto1(){
      this.$router.push({path:"/login"});
    },
    goto2(){
      this.$router.push({path:"/generate"});
    }
  }
}
</script>
<style>
.gobuttonBox{
  position:absolute;
  height: 10vh;
  top:60%;
  left: 50%;
  transform: translate(-50%,-50%);
}
</style>
