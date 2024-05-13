from .learner_module.ppo.learning import learning as alearning_ppo


def ppo_awrapper(timer):
    def _outer(func):  # 주의) func 자체는 껍데기 동기함수
        async def _inner(self, *args, **kwargs):
            return await alearning_ppo(self, timer, *args, **kwargs)

        return _inner

    return _outer