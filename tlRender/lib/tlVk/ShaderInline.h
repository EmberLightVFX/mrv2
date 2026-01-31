// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025-Present Gonzalo Garramuño
// All rights reserved.

#include <FL/Fl_Vk_Utils.H>

namespace tl
{
    namespace vlk
    {

        template <typename T>
        void Shader::createUniform(
            const std::string& name, const T& value,
            const ShaderFlags stageFlags)
        {
            auto it = ubos.find(name);
            if (it != ubos.end())
            {
                throw std::runtime_error(
                    name + " for shader " + shaderName + " already created.");
            }

            UBOBinding ubo;
            ubo.size = sizeof(value);

            ubo.layoutBinding.binding = current_binding_index++;
            ubo.layoutBinding.descriptorCount = 1;
            ubo.layoutBinding.descriptorType =
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            ubo.layoutBinding.stageFlags = getVulkanShaderFlags(stageFlags);
            ubo.layoutBinding.pImmutableSamplers = nullptr;

            ubos[name] = ubo;
        }

        void Shader::createUniformData(
            const std::string& name, const size_t size,
            const ShaderFlags stageFlags)
        {
            auto it = ubos.find(name);
            if (it != ubos.end())
            {
                throw std::runtime_error(
                    name + " for shader " + shaderName + " already created.");
            }

            UBOBinding ubo;
            ubo.size = size;

            ubo.layoutBinding.binding = current_binding_index++;
            ubo.layoutBinding.descriptorCount = 1;
            ubo.layoutBinding.descriptorType =
                VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            ubo.layoutBinding.stageFlags = getVulkanShaderFlags(stageFlags);
            ubo.layoutBinding.pImmutableSamplers = nullptr;

            ubos[name] = ubo;
        }

        
        template <typename T>
        void Shader::setUniform(
            const std::string& name, const T& value,
            const ShaderFlags stageFlags)
        {
            if (!activeBindingSet)
                throw std::runtime_error("No activeBindingSet for Shader " + name);
            activeBindingSet->updateUniform(name, &value, sizeof(value), frameIndex);
        }
        
        void Shader::setUniformData(
            const std::string& name, const void* data, const size_t size,
            const ShaderFlags stageFlags)
        {
            if (!activeBindingSet)
                throw std::runtime_error("No activeBindingSet for Shader " + name);
            activeBindingSet->updateUniform(name, data, size, frameIndex);
        }

        template <typename T>
        void Shader::addPush(
            const std::string& name, const T& value,
            const ShaderFlags stageFlags)
        {
            pushSize = sizeof(T);
            pushStageFlags = getVulkanShaderFlags(stageFlags);
        }

    } // namespace vlk
} // namespace tl
