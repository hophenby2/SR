SamplerState screen_texture_sampler : register(s4);
Texture2D screen_texture : register(t4);

cbuffer engine_data : register(b1)
{
    float4 screen_texture_size;
    float4 viewport;
};

cbuffer user_data : register(b0)
{
    float4 effect_param;
};

struct PS_Input
{
    float4 sxy : SV_Position;
    float2 uv : TEXCOORD0;
    float4 col : COLOR0;
};

float4 main(PS_Input input) : SV_Target
{
    float2 texture_size = screen_texture_size.xy;
    float2 xy = input.uv * texture_size;
    float2 center = effect_param.xy;
    float angle = radians(effect_param.z);
    float axis_tilt = effect_param.w;
    float2 rotation = float2(cos(angle), sin(angle));
    float wave = radians((xy.x - xy.y * axis_tilt) * 0.9);
    float wave_value = sin(wave);
    float wave_squared = wave_value * wave_value;
    float2 scale = float2(1.0 + 0.04 * wave_squared, 1.0 + 0.075 * wave_squared * wave_squared);
    float2 transformed = xy * scale;
    float2 delta = (transformed - center) * 0.5;
    float2 sample_xy = float2(
        delta.x * rotation.x - delta.y * rotation.y,
        delta.x * rotation.y + delta.y * rotation.x) + center;
    float2 sample_uv = sample_xy / texture_size;

    if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0)
    {
        sample_uv = input.uv;
    }

    float4 color = screen_texture.Sample(screen_texture_sampler, sample_uv);
    color.a = 1.0;
    return color;
}
