// TODO Rewrite using vulkan.hpp
// TODO Implement runtime GLSL to SPIR-V

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <string>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <algorithm>
#include <set>
#include <fstream>

const int WIDTH = 800;
const int HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_LUNARG_standard_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

bool checkValidationLayerSupport() {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {
        auto pred = [&](VkLayerProperties x) { return !strcmp(layerName, x.layerName); };
        if (std::none_of(availableLayers.begin(), availableLayers.end(), pred)) {
            return false;
        }
    }

    return true;
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pCallback) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pCallback);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT callback, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, callback, pAllocator);
    }
}

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
        throw std::runtime_error("Failed to open file!");

    size_t fileSize = (size_t) file.tellg();
    std::vector<char> buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();
    return buffer;
}

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapchainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class HelloTriangleApplication {
    public:
        void run() {
            initWindow();
            initVulkan();
            mainLoop();
            cleanup();
        }
    private:
        GLFWwindow* m_window;

        VkInstance m_instance;

        VkDebugUtilsMessengerEXT m_callback;

        VkSurfaceKHR m_surface;

        VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
        VkDevice m_device;

        VkQueue m_graphicsQueue;
        VkQueue m_presentQueue;

        VkSwapchainKHR m_swapchain;

        std::vector<VkImage> m_swapchainImages;
        VkFormat m_swapchainImageFormat;
        VkExtent2D m_swapchainExtent;

        std::vector<VkImageView> m_swapchainImageViews;

        VkRenderPass m_renderPass;
        VkPipelineLayout m_pipelineLayout;

        VkPipeline m_graphicsPipeline;

        std::vector<VkFramebuffer> m_swapchainFramebuffers;

        VkCommandPool m_commandPool;
        std::vector<VkCommandBuffer> m_commandBuffers;

        std::vector<VkSemaphore> m_imageAvailableSemaphores;
        std::vector<VkSemaphore> m_renderFinishedSemaphores;
        std::vector<VkFence> m_inFlightFences;
        size_t m_currentFrame = 0;

        bool framebufferResized = false;

        void initWindow() {
            glfwInit();

            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

            m_window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
            glfwSetWindowUserPointer(m_window, this);
            glfwSetFramebufferSizeCallback(m_window, framebufferResizeCallback);
        }

        static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
            auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
            app->framebufferResized = true;
        }

        void initVulkan() {
            createInstance();
            setupDebugCallback();
            createSurface();
            pickPhysicalDevice();
            createLogicalDevice();
            createSwapchain();
            createImageViews();
            createRenderPass();
            createGraphicsPipeline();
            createFramebuffers();
            createCommandPool();
            createCommandBuffers();
            createSyncObjects();
        }

        void cleanupSwapchain() {
            for (auto framebuffer : m_swapchainFramebuffers) {
                vkDestroyFramebuffer(m_device, framebuffer, nullptr);
            }

            vkFreeCommandBuffers(m_device, m_commandPool, static_cast<uint32_t>(m_commandBuffers.size()),  m_commandBuffers.data());

            vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
            vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
            vkDestroyRenderPass(m_device, m_renderPass, nullptr);

            for (auto imageView : m_swapchainImageViews) {
                vkDestroyImageView(m_device, imageView, nullptr);
            }

            vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
        }

        void recreateSwapchain() {
            int width = 0, height = 0;
            while (width == 0 || height == 0) {
                glfwGetFramebufferSize(m_window, &width, &height);
                glfwWaitEvents();
            }

            vkDeviceWaitIdle(m_device);

            cleanupSwapchain();

            createSwapchain();
            createImageViews();
            createRenderPass();
            createGraphicsPipeline();
            createFramebuffers();
            createCommandBuffers();
        }

        void createSyncObjects() {
            m_imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
            m_renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
            m_inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

            VkSemaphoreCreateInfo semaphoreInfo = {};
            semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

            VkFenceCreateInfo fenceInfo = {};
            fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(m_device, &fenceInfo, nullptr, &m_inFlightFences[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to create sync objects for a frame!");

            }
        }

        void createCommandBuffers() {
            m_commandBuffers.resize(m_swapchainFramebuffers.size());

            VkCommandBufferAllocateInfo allocInfo = {};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = m_commandPool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = (uint32_t) m_commandBuffers.size();

            if (vkAllocateCommandBuffers(m_device, &allocInfo, m_commandBuffers.data()) != VK_SUCCESS)
                throw std::runtime_error("Failed to allocate command buffers!");

            for (size_t i = 0; i < m_commandBuffers.size(); i++) {
                VkCommandBufferBeginInfo beginInfo = {};
                beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
                beginInfo.pInheritanceInfo = nullptr; // Optional

                if (vkBeginCommandBuffer(m_commandBuffers[i], &beginInfo) != VK_SUCCESS)
                    throw std::runtime_error("Failed to begin recording command buffer!");

                VkRenderPassBeginInfo renderPassInfo = {};
                renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                renderPassInfo.renderPass = m_renderPass;
                renderPassInfo.framebuffer = m_swapchainFramebuffers[i];

                renderPassInfo.renderArea.offset = {0, 0};
                renderPassInfo.renderArea.extent = m_swapchainExtent;

                // Try 0.0f instead of 1.0f
                VkClearValue clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
                renderPassInfo.clearValueCount = 1;
                renderPassInfo.pClearValues = &clearColor;

                vkCmdBeginRenderPass(m_commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
                vkCmdBindPipeline(m_commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
                vkCmdDraw(m_commandBuffers[i], 3, 1, 0, 0);
                vkCmdEndRenderPass(m_commandBuffers[i]);

                if (vkEndCommandBuffer(m_commandBuffers[i]) != VK_SUCCESS)
                    throw std::runtime_error("Failed to record command buffer!");
            }
        }

        void createCommandPool() {
            QueueFamilyIndices queueFamilyIndices = findQueueFamilies(m_physicalDevice);

            VkCommandPoolCreateInfo poolInfo = {};
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
            poolInfo.flags = 0; // Optional

            if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool) != VK_SUCCESS)
                throw std::runtime_error("Failed to create command pool!");
        }

        void createFramebuffers() {
            m_swapchainFramebuffers.resize(m_swapchainImageViews.size());

            for (size_t i = 0; i < m_swapchainImageViews.size(); i++) {
                VkImageView attachments[] = {
                    m_swapchainImageViews[i]
                };

                VkFramebufferCreateInfo framebufferInfo = {};
                framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebufferInfo.renderPass = m_renderPass;
                framebufferInfo.attachmentCount = 1;
                framebufferInfo.pAttachments = attachments;
                framebufferInfo.width = m_swapchainExtent.width;
                framebufferInfo.height = m_swapchainExtent.height;
                framebufferInfo.layers = 1;

                if (vkCreateFramebuffer(m_device, &framebufferInfo, nullptr, &m_swapchainFramebuffers[i]) != VK_SUCCESS)
                    throw std::runtime_error("Failed to create framebuffer!");
            }
        }

        void createRenderPass() {
            VkAttachmentDescription colorAttachment = {};
            colorAttachment.format = m_swapchainImageFormat;
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachment.finalLayout  = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

            VkAttachmentReference colorAttachmentRef = {};
            colorAttachmentRef.attachment = 0;
            colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkSubpassDescription subpass = {};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &colorAttachmentRef;

            VkSubpassDependency dependency = {};
            dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
            dependency.dstSubpass = 0;

            dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.srcAccessMask = 0;

            dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

            VkRenderPassCreateInfo renderPassInfo = {};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassInfo.attachmentCount = 1;
            renderPassInfo.pAttachments = &colorAttachment;
            renderPassInfo.subpassCount = 1;
            renderPassInfo.pSubpasses = &subpass;

            renderPassInfo.dependencyCount = 1;
            renderPassInfo.pDependencies = &dependency;

            if (vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_renderPass) != VK_SUCCESS)
                throw std::runtime_error("Failed to create render pass!");
        }

        void createGraphicsPipeline() {
            auto vertShaderCode = readFile("shaders/vert.spv");
            auto fragShaderCode = readFile("shaders/frag.spv");

            VkShaderModule vertShaderModule;
            VkShaderModule fragShaderModule;

            vertShaderModule = createShaderModule(vertShaderCode);
            fragShaderModule = createShaderModule(fragShaderCode);

            VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
            vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
            vertShaderStageInfo.module = vertShaderModule;
            vertShaderStageInfo.pName = "main";

            VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
            fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            fragShaderStageInfo.module = fragShaderModule;
            fragShaderStageInfo.pName = "main";

            VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

            VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
            vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

            vertexInputInfo.vertexBindingDescriptionCount = 0;
            vertexInputInfo.pVertexBindingDescriptions = nullptr;
            vertexInputInfo.vertexAttributeDescriptionCount = 0;
            vertexInputInfo.pVertexAttributeDescriptions = nullptr;

            VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
            inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            inputAssembly.primitiveRestartEnable = VK_FALSE;

            VkViewport viewport = {};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = (float) m_swapchainExtent.width;
            viewport.height = (float) m_swapchainExtent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;

            VkRect2D scissor = {};
            scissor.offset = {0, 0};
            scissor.extent = m_swapchainExtent;

            VkPipelineViewportStateCreateInfo viewportState = {};
            viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewportState.viewportCount = 1;
            viewportState.pViewports = &viewport;
            viewportState.scissorCount = 1;
            viewportState.pScissors = &scissor;

            VkPipelineRasterizationStateCreateInfo rasterizer = {};
            rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterizer.depthClampEnable = VK_FALSE;
            rasterizer.rasterizerDiscardEnable = VK_FALSE;
            rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
            rasterizer.lineWidth = 1.0f;
            rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
            rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

            rasterizer.depthBiasEnable = VK_FALSE;
            rasterizer.depthBiasConstantFactor = 0.0f;
            rasterizer.depthBiasClamp = 0.0f;
            rasterizer.depthBiasSlopeFactor = 0.0f;

            VkPipelineMultisampleStateCreateInfo multisampling = {};
            multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisampling.sampleShadingEnable = VK_FALSE;
            multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            multisampling.minSampleShading = 1.0f;
            multisampling.pSampleMask = nullptr;
            multisampling.alphaToCoverageEnable = VK_FALSE;
            multisampling.alphaToOneEnable = VK_FALSE;

            VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
            colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            colorBlendAttachment.blendEnable = VK_FALSE;

            // All of these are optional
            colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
            colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

            VkPipelineColorBlendStateCreateInfo colorBlending = {};
            colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            colorBlending.logicOpEnable = VK_FALSE;
            colorBlending.logicOp = VK_LOGIC_OP_COPY;
            colorBlending.attachmentCount = 1;
            colorBlending.pAttachments = &colorBlendAttachment;
            
            // Optional
            colorBlending.blendConstants[0] = 0.0f;
            colorBlending.blendConstants[1] = 0.0f;
            colorBlending.blendConstants[2] = 0.0f;
            colorBlending.blendConstants[3] = 0.0f;

            VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

            // Optional
            pipelineLayoutInfo.setLayoutCount = 0;
            pipelineLayoutInfo.pSetLayouts = nullptr;
            pipelineLayoutInfo.pushConstantRangeCount = 0;
            pipelineLayoutInfo.pPushConstantRanges = nullptr;

            if (vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS)
                throw std::runtime_error("Failed to create pipeline layout!");

            VkGraphicsPipelineCreateInfo pipelineInfo = {};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo.stageCount = 2;
            pipelineInfo.pStages = shaderStages;

            pipelineInfo.pVertexInputState = &vertexInputInfo;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &rasterizer;
            pipelineInfo.pMultisampleState = &multisampling;
            pipelineInfo.pDepthStencilState = nullptr;
            pipelineInfo.pColorBlendState = &colorBlending;
            pipelineInfo.pDynamicState = nullptr;

            pipelineInfo.layout = m_pipelineLayout;

            pipelineInfo.renderPass = m_renderPass;
            pipelineInfo.subpass = 0;

            // Optional
            pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
            pipelineInfo.basePipelineIndex = -1;

            if (vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_graphicsPipeline) != VK_SUCCESS)
                throw std::runtime_error("Failed to create graphics pipeline!");

            vkDestroyShaderModule(m_device, fragShaderModule, nullptr);
            vkDestroyShaderModule(m_device, vertShaderModule, nullptr);
        }

        VkShaderModule createShaderModule(const std::vector<char>& code) {
            VkShaderModuleCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            createInfo.codeSize = code.size();
            createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

            VkShaderModule shaderModule;
            if (vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create shader module!");
            }

            return shaderModule;
        }

        void createImageViews() {
            m_swapchainImageViews.resize(m_swapchainImages.size());
            for (size_t i = 0; i < m_swapchainImages.size(); i++) {
                VkImageViewCreateInfo createInfo = {};
                createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                createInfo.image = m_swapchainImages[i];

                createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
                createInfo.format = m_swapchainImageFormat;

                createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
                createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

                createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                createInfo.subresourceRange.baseMipLevel = 0;
                createInfo.subresourceRange.levelCount = 1;
                createInfo.subresourceRange.baseArrayLayer = 0;
                createInfo.subresourceRange.layerCount = 1;

                if (vkCreateImageView(m_device, &createInfo, nullptr, &m_swapchainImageViews[i]) != VK_SUCCESS) {
                        throw std::runtime_error("Failed to create image views!");
                }
            }
        }

        void createInstance() {

            uint32_t availableExtensionCount = 0;
            vkEnumerateInstanceExtensionProperties(nullptr, &availableExtensionCount, nullptr);

            std::vector<VkExtensionProperties> availableExtensions(availableExtensionCount);
            vkEnumerateInstanceExtensionProperties(nullptr, &availableExtensionCount, availableExtensions.data());

            auto requiredExtensions = getRequiredExtensions();
            uint32_t requiredExtensionCount = requiredExtensions.size();

            if (requiredExtensionCount == 0 || (requiredExtensionCount == 1 && enableValidationLayers)) {
                throw std::runtime_error("Failed to get extensions required by GLFW");
            } else {
                // Check if we are missing any extensions required by glfw
                std::string missingExtensions = "";
                for (int i = 0; i < requiredExtensionCount; i++) {
                    auto pred = [&](VkExtensionProperties x) { return !strcmp(x.extensionName, requiredExtensions[i]); };
                    if ( std::none_of(availableExtensions.begin(), availableExtensions.end(), pred)) { 
                        missingExtensions+= "\t";
                        missingExtensions+= requiredExtensions[i];
                        missingExtensions+= "\n";
                    }
                }

                if (missingExtensions != "") {
                    throw std::runtime_error("Missing extensions required by GLFW: \n" + missingExtensions);
                }
            }

            VkApplicationInfo appInfo = {};
            appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            appInfo.pApplicationName = "Hello Triangle";
            appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.pEngineName = "No Engine";
            appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
            appInfo.apiVersion = VK_API_VERSION_1_1;

            VkInstanceCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            createInfo.pApplicationInfo = &appInfo;

            createInfo.enabledExtensionCount = requiredExtensionCount;
            createInfo.ppEnabledExtensionNames = requiredExtensions.data();

            if (enableValidationLayers) {
                if (!checkValidationLayerSupport())
                    throw std::runtime_error("Validation layers requested, but not available!");
                createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
                createInfo.ppEnabledLayerNames = validationLayers.data();
            } else {
                /* createInfo.enabledLayerCount = 0; */
                throw std::runtime_error("Didn't enable validation layers!");
            }

            if (vkCreateInstance(&createInfo, nullptr, &m_instance) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create instance!");
            }
        }

        void createSurface() {
            if (glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create window surface!");
            }
        }

        void setupDebugCallback() {
            if (!enableValidationLayers) return;
            VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;

            createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;

            createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT    |
                                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;

            createInfo.pfnUserCallback = debugCallback;
            createInfo.pUserData = nullptr; // Optional

            if (CreateDebugUtilsMessengerEXT(m_instance, &createInfo, nullptr, &m_callback) != VK_SUCCESS) {
                throw std::runtime_error("Failed to setup debug callback!");
            }
        }

        void pickPhysicalDevice() {
            uint32_t deviceCount = 0;
            vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);

            if (!deviceCount)
                throw std::runtime_error("Failed to find GPUs with Vulkan support!");

            std::vector<VkPhysicalDevice> devices(deviceCount);
            vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());

            for (const auto& device : devices) {
                if (isDeviceSuitable(device)) {
                    m_physicalDevice = device;
                    break;
                }
            }

            if (m_physicalDevice == VK_NULL_HANDLE)
                throw std::runtime_error("Failed to find a suitable GPU!");
        }

        void createLogicalDevice() {
            QueueFamilyIndices indices = findQueueFamilies(m_physicalDevice);

            std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
            std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

            float queuePriority = 1.0f;
            for (uint32_t queueFamily : uniqueQueueFamilies) {
                VkDeviceQueueCreateInfo queueCreateInfo = {};
                queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                queueCreateInfo.queueFamilyIndex = queueFamily;
                queueCreateInfo.queueCount = 1;
                queueCreateInfo.pQueuePriorities = &queuePriority;
                queueCreateInfos.push_back(queueCreateInfo);
            }

            VkPhysicalDeviceFeatures deviceFeatures = {};
            VkDeviceCreateInfo createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

            createInfo.pQueueCreateInfos = queueCreateInfos.data();
            createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());

            createInfo.pEnabledFeatures = &deviceFeatures;

            createInfo.ppEnabledExtensionNames = deviceExtensions.data();
            createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());

            if (enableValidationLayers) {
                createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
                createInfo.ppEnabledLayerNames = validationLayers.data();
            } else {
                createInfo.enabledLayerCount = 0;
            }

            if (vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device) != VK_SUCCESS)
                throw std::runtime_error("Failed to create logical device!");

            vkGetDeviceQueue(m_device, indices.graphicsFamily.value(), 0, &m_graphicsQueue);
            vkGetDeviceQueue(m_device, indices.presentFamily.value(), 0, &m_presentQueue);
        }

        bool isDeviceSuitable(VkPhysicalDevice device) {
            QueueFamilyIndices indices = findQueueFamilies(device);

            bool extensionsSupported = checkDeviceExtensionSupport(device);

            bool swapchainAdequate = false;
            if (extensionsSupported) {
                SwapchainSupportDetails swapchainSupport = querySwapchainSupport(device);
                swapchainAdequate = !swapchainSupport.formats.empty() && !swapchainSupport.presentModes.empty();
            }

            return indices.isComplete() && extensionsSupported && swapchainAdequate;
        }

        VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
            if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED) {
                return { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
            }

            for (const auto& availableFormat : availableFormats) {
                if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                    return availableFormat;
                }
            }

            return availableFormats[0];
        }

        // Caps fps to 60
        // CPU load decreases < 1%
        VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> availablePresentModes) {
            return VK_PRESENT_MODE_FIFO_KHR;

            /* VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR; */

            /* for (const auto& availablePresentMode : availablePresentModes) { */
            /*     if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) { */
            /*         std::cout << "Present mode chosen: " << VK_PRESENT_MODE_MAILBOX_KHR; */
            /*         return availablePresentMode; */
            /*     } else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) { */
            /*         bestMode = availablePresentMode; */
            /*     } */
            /* } */

            /* return bestMode; */
        }

        VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
            if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
                return capabilities.currentExtent;
            } else {
                int width, height;
                glfwGetFramebufferSize(m_window, &width, &height);

                VkExtent2D actualExtent = {
                    static_cast<uint32_t>(width),
                    static_cast<uint32_t>(height)
                };

                actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
                actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

                return actualExtent;
            }
        }

        bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
            uint32_t extensionCount;
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

            std::vector<VkExtensionProperties> availableExtensions(extensionCount);
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

            std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

            for (const auto& extension : availableExtensions) {
                requiredExtensions.erase(extension.extensionName);
            }

            return requiredExtensions.empty();
        }

        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
            QueueFamilyIndices indices;

            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

            std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

            int i = 0;
            for (const auto& queueFamily : queueFamilies) {
                VkBool32 presentSupport = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(device, i, m_surface, &presentSupport);

                if (queueFamily.queueCount > 0 && queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
                    indices.graphicsFamily = i;

                if (queueFamily.queueCount > 0 && presentSupport)
                    indices.presentFamily = i;

                if (indices.isComplete())
                    break;

                i++;
            }

            return indices;
        }

        std::vector<const char*> getRequiredExtensions() {
            uint32_t glfwExtensionCount = 0;
            const char** glfwExtensions;
            glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
            std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

            if (enableValidationLayers)
                extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

            return extensions;
        }

        SwapchainSupportDetails querySwapchainSupport(VkPhysicalDevice device) {
            SwapchainSupportDetails details;

            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, m_surface, &details.capabilities);

            uint32_t formatCount;
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &formatCount, nullptr);

            if (formatCount != 0) {
                details.formats.resize(formatCount);
                vkGetPhysicalDeviceSurfaceFormatsKHR(device, m_surface, &formatCount, details.formats.data());
            }

            uint32_t presentModeCount;
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_surface, &presentModeCount, nullptr);

            if (presentModeCount != 0) {
                details.presentModes.resize(presentModeCount);
                vkGetPhysicalDeviceSurfacePresentModesKHR(device, m_surface, &presentModeCount, details.presentModes.data());
            }

            return details;
        }

        void createSwapchain() {
            SwapchainSupportDetails swapchainSupport = querySwapchainSupport(m_physicalDevice);
            VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapchainSupport.formats);
            VkPresentModeKHR presentMode = chooseSwapPresentMode(swapchainSupport.presentModes);
            VkExtent2D extent = chooseSwapExtent(swapchainSupport.capabilities);

            // Adding one more image to properly implement triple buffering
            uint32_t imageCount = swapchainSupport.capabilities.minImageCount + 1;

            if (swapchainSupport.capabilities.maxImageCount > 0 && imageCount > swapchainSupport.capabilities.maxImageCount) {
                imageCount = swapchainSupport.capabilities.maxImageCount;
            }

            VkSwapchainCreateInfoKHR createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
            createInfo.surface = m_surface;
            createInfo.minImageCount = imageCount;
            createInfo.imageFormat = surfaceFormat.format;
            createInfo.imageColorSpace = surfaceFormat.colorSpace;
            createInfo.imageExtent = extent;
            createInfo.imageArrayLayers = 1;
            createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

            QueueFamilyIndices indices = findQueueFamilies(m_physicalDevice);
            uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

            if (indices.graphicsFamily != indices.presentFamily) {
                createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
                createInfo.queueFamilyIndexCount = 2;
                createInfo.pQueueFamilyIndices = queueFamilyIndices;
            } else {
                createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
                createInfo.queueFamilyIndexCount = 0; // Optional
                createInfo.pQueueFamilyIndices = nullptr; // Optional
            }

            createInfo.preTransform = swapchainSupport.capabilities.currentTransform;
            createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

            createInfo.presentMode = presentMode;
            createInfo.clipped = VK_TRUE;

            createInfo.oldSwapchain = VK_NULL_HANDLE;

            if (vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapchain) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create swapchain!");
            }

            vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, nullptr);
            m_swapchainImages.resize(imageCount);
            vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, m_swapchainImages.data());

            m_swapchainImageFormat = surfaceFormat.format;
            m_swapchainExtent = extent;
        }

        void mainLoop() {
            while (!glfwWindowShouldClose(m_window)) {
                glfwPollEvents();
                drawFrame();
            }

            vkDeviceWaitIdle(m_device);
        }

        void drawFrame() {
            vkWaitForFences(m_device,1,&m_inFlightFences[m_currentFrame],VK_TRUE,std::numeric_limits<uint64_t>::max());

            uint32_t imageIndex;
            VkResult result = vkAcquireNextImageKHR(m_device, m_swapchain, std::numeric_limits<uint64_t>::max(), m_imageAvailableSemaphores[m_currentFrame], VK_NULL_HANDLE, &imageIndex);

            if (result == VK_ERROR_OUT_OF_DATE_KHR) {
                recreateSwapchain();
                return;
            } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
                throw std::runtime_error("Failed to acquire swapchain image!");

            
            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

            VkSemaphore waitSemaphores[] = {m_imageAvailableSemaphores[m_currentFrame]};
            VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = waitSemaphores;
            submitInfo.pWaitDstStageMask = waitStages;

            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &m_commandBuffers[imageIndex];

            VkSemaphore signalSemaphores[] = {m_renderFinishedSemaphores[m_currentFrame]};
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = signalSemaphores;

            vkResetFences(m_device, 1, &m_inFlightFences[m_currentFrame]);

            if (vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, m_inFlightFences[m_currentFrame]) != VK_SUCCESS)
                throw std::runtime_error("Failed to submit draw command buffer!");

            VkPresentInfoKHR presentInfo = {};
            presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = signalSemaphores;

            VkSwapchainKHR swapchains[] = {m_swapchain};
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = swapchains;
            presentInfo.pImageIndices = &imageIndex;

            presentInfo.pResults = nullptr; // Optional
            result = vkQueuePresentKHR(m_presentQueue, &presentInfo);

            if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
                framebufferResized = false;
                recreateSwapchain();
            } else if (result != VK_SUCCESS)
                throw std::runtime_error("Failed to present swapchain image!");

            m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        }

        void cleanup() {
            cleanupSwapchain();

            for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
                vkDestroySemaphore(m_device, m_renderFinishedSemaphores[i], nullptr);
                vkDestroySemaphore(m_device, m_imageAvailableSemaphores[i], nullptr);
                vkDestroyFence(m_device, m_inFlightFences[i], nullptr);
            }

            vkDestroyCommandPool(m_device, m_commandPool, nullptr);
            
            vkDestroyDevice(m_device, nullptr);

            if (enableValidationLayers)
                DestroyDebugUtilsMessengerEXT(m_instance, m_callback, nullptr);

            vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
            vkDestroyInstance(m_instance, nullptr);

            glfwDestroyWindow(m_window);

            glfwTerminate();
        }

        static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
            VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
            VkDebugUtilsMessageTypeFlagsEXT messageType,
            const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
            void* pUserData) {

            std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

            return VK_FALSE;
        }
};

int main() {
    HelloTriangleApplication v_app;

    try {
        v_app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
