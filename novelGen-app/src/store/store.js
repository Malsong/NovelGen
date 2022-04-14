import { createStore } from 'vuex'


export default createStore({
    state () {
        return {
            user: undefined,
            stateOn:false
        }
    },
    mutations: {    //同步修改
        login (state,userInfo) {
            state.user = userInfo
            state.stateOn=true
        },
        logOut(state){
            state.user = undefined
            state.stateOn=false
        }
    },
    actions:{   //异步修改：actions不能直接修改全局变量，需要调用commit方法来触发mutation中的方法
        login (context, userInfo) {
            context.commit('login', userInfo)
        },
        logOut (context) {
            context.commit('logOut')
        }
    }
})
  


//修改store中的值唯一的方法就是提交mutation来修改
/* export default new Vuex.Store({
    state:{ //要设置的全局访问的state对象
        count:0, //this.$store.state.count 可以拿到 state里面的 count
        change:0
    },
    getters:{ //实时监听state值的变化（最新状态）
        getCount(state){ //承载变化的count值
            return state.count  //this.$store.getters.getCount通过 getters 获取 承载变化的 count 的值
        }
    },
    mutations:{
        //可通过vue的methods commit方法进行改变state中的参数
    },
    actions:{
        //提交的为mutation，不直接改变状态
        //可包含任意异步操作 vue中commit变为dispatch来提交action
    }
}) */