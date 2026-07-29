SamplerState screen_texture_sampler : register(s4);
Texture2D screen_texture : register(t4);

cbuffer engine_data : register(b1)
{
    float4 screen_texture_size;
    float4 viewport;
};

cbuffer user_data : register(b0)
{
    float4 center_pos;
    float4 effect_param;
};

struct PS_Input
{
    float4 sxy : SV_Position;
    float2 uv : TEXCOORD0;
    float4 col : COLOR0;
};

float2 distortion(float2 xy, float delta_len)
{
    float effect_size = effect_param.x;
    float effect_arg = effect_param.y;
    float timer = effect_param.z;
    float k = delta_len / effect_size;
    float p = (k - 1.0) * (k - 1.0);
    float2 delta1 = float2(effect_arg * 0.8 * sin((xy.x * 0.5 + xy.y) / 18.0 + timer / 5.0), 0.0);
    float arg = lerp(effect_arg * 1.2, effect_arg * 0.8, sin(timer / 10.0) * 0.5 + 1.0);
    float2 delta2 = delta_len * (1.0 - lerp(pow(k, 1.0 + arg), k, k));
    return delta1 * p + delta2 * p * 0.8;
}

float4 main(PS_Input input) : SV_Target
{
    float2 texture_size = screen_texture_size.xy;
    float2 xy = input.uv * texture_size;
    float2 delta = xy - center_pos.xy;
    float delta_len = length(delta);
    float2 sample_uv = input.uv;

    if (delta_len <= effect_param.x && effect_param.x > 0.0)
    {
        float2 offset = distortion(xy, delta_len);
        float2 candidate = input.uv + offset / texture_size;
        if (candidate.x >= 0.0 && candidate.x <= 1.0 && candidate.y >= 0.0 && candidate.y <= 1.0)
        {
            sample_uv = candidate;
        }
    }

    float4 color = screen_texture.Sample(screen_texture_sampler, sample_uv);
    color.a = 1.0;
    return color;
}
