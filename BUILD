deps = select({
    "@platforms//cpu:x86_64": ["@mujoco_amd64//:mujoco"],
    "@platforms//cpu:arm64": ["@mujoco_arm64//:mujoco"],
    "//conditions:default": ["@platforms//:incompatible"],
})

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = deps,
    visibility = ["//visibility:public"],
)