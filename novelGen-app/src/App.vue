<template>
    <router-view></router-view>
</template>

<script>
    import Start from './startpage/Start.vue'
    /* import BacePage from './layout/BasePage.vue'
    import LoginRegister from './components/LoginRegister.vue' */
    export default {
        name:"app",
        components:{
            Start
        }
    }
</script>
<style>
*{
    /* 初始化 */
    margin: 0;
    padding: 0;
}
body{
    /* 渐变背景 */
    background: linear-gradient(200deg,#f3e7e9,#e3eeff);
    background-image:url(../src/assets/bg_true.png) ;
    background-size:cover;
    /* background-size:90% 100%; */
    /* background-size:100vh 100vw ; */
}
</style>
