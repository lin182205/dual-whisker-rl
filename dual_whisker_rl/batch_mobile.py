"""办公批量实验共用的混合策略构建和模型契约。"""
import json
from dual_whisker_rl.mobile_contract import environment_contract
from dual_whisker_rl.mobile_features import feature_layout_kwargs


def policy_setup(env, history_length, encoder='gru', width=160):
    from dual_whisker_rl.agents.hybrid_policy import HybridActorCriticPolicy
    from dual_whisker_rl.agents.olfactory_navigation import OlfactoryNavigationExtractor
    return HybridActorCriticPolicy, dict(net_arch=[width,width],share_features_extractor=True,
        features_extractor_class=OlfactoryNavigationExtractor,
        features_extractor_kwargs=dict(history_length=history_length,base_observation_dim=57,
                                       temporal_encoder=encoder,**feature_layout_kwargs(env.unwrapped.effective_config)))


def model_contract(env, variant):
    result=environment_contract(env.unwrapped,int(variant.get('history_length',20)))
    space=env.action_space
    result['batch_policy_action']={'shape':list(space.shape),'low':space.low.tolist(),'high':space.high.tolist(),
                                   'categorical_branches':getattr(space,'categorical_branches',2)}
    result['batch_variant']=variant
    result['batch_version']='office_batch_v1'
    return json.loads(json.dumps(result,sort_keys=True))


def attach_contract(model, env, variant):
    contract=model_contract(env,variant)
    contract['encoder_config']=model.policy.features_extractor.configuration
    model.batch_environment_contract=contract
    return contract


def validate_contract(model, env, variant, *, resume=False):
    from dual_whisker_rl.agents.hybrid_policy import HybridActorCriticPolicy
    if not isinstance(model.policy,HybridActorCriticPolicy):
        raise ValueError('办公批量实验需要联合高斯/分类策略')
    expected=model_contract(env,variant)
    expected['encoder_config']=model.policy.features_extractor.configuration
    if expected['encoder_config'].get('feature_layout') != expected.get('feature_layout'):
        raise ValueError('办公批量模型的历史分支契约与配置不一致')
    stored=getattr(model,'batch_environment_contract',None)
    if stored is None:
        raise ValueError('缺少办公批量实验契约，不能复用旧地图或独立实验模型')
    actual=json.loads(json.dumps(stored))
    if not resume:
        allowed=['domain_randomization','record_trajectory']
        if getattr(env.unwrapped,'fault',{}):
            allowed+=['servo_60deg_time_s','sensor_response_tau','sensor_recovery_tau','sensor_noise_std']
        for key in allowed:
            actual['effective_config'].pop(key,None);expected['effective_config'].pop(key,None)
    if actual!=expected or model.observation_space!=env.observation_space or model.action_space!=env.action_space:
        raise ValueError('办公批量模型契约不匹配（地图、字段、动作、消融或物理配置）')
