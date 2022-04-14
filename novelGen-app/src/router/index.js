
/* import Vue from 'vue'; */
import { createRouter,createWebHashHistory} from 'vue-router';
import Start from '../startpage/Start.vue';
import RegisLoginRegisterter from '../components/LoginRegister.vue';
import BasePage from '../layout/BasePage.vue'
import GenPage from '../components/GenPage.vue'
import SavePage from '../components/SavePage.vue'
/* import NavHeader from '../components/NavHeader.vue' */
//单个路由均为对象类型，path代表的是路径，component代表组件
const routes=[
    {
        path:'/',
        name:'start',
        component: Start
    },
    {
        path:'/login',
        name:'login',
        component: RegisLoginRegisterter
    },
    {
        path:'/base',
        name:'base',
        component: BasePage,
        children:[
            {
                path:'/generate',
                name:'generate',
                component: GenPage
            },
            {
                path:'/save',
                name:'save',
                component: SavePage,
                meta:{
                    stateOn:true
                }
            }
        ]
    },
]


//实例化VueRouter并将routes添加进去
const router=createRouter({
    history:createWebHashHistory(),  /* 路由模式 */
    routes:routes
});

router.beforeEach((to, from, next) => {
    if(to.path.indexOf('userlogout') !== -1){
      
    }
    else if(to.path.indexOf('backtohome') !== -1){

    }
    else if(to.path.indexOf('getuserInfo')!==-1){}
    else if(to.path.indexOf('changeInfo')!==-1){}
    else{
        next();
    }
  })


//抛出这个这个实例对象方便外部读取以及访问

export default router