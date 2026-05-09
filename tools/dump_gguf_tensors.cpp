// Quick CLI: dump tensor names from a GGUF file. Build with the same nt_infra link.
#include <model/loader.h>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <gguf_path>\n", argv[0]);
        return 1;
    }
    nt::GGUFLoader loader;
    if (!loader.load(argv[1])) {
        std::fprintf(stderr, "failed to load: %s\n", argv[1]);
        return 1;
    }
    auto names = loader.tensor_names();
    std::printf("# %zu tensors in %s\n", names.size(), argv[1]);
    for (const auto& n : names) {
        const auto* info = loader.tensor_info(n);
        if (!info) continue;
        std::string shape_str;
        for (size_t i = 0; i < info->shape.size(); ++i) {
            if (i) shape_str += ",";
            shape_str += std::to_string(info->shape[i]);
        }
        std::printf("%-40s ggml_type=%d shape=[%s]\n", n.c_str(),
                    static_cast<int>(info->ggml_type), shape_str.c_str());
    }
    return 0;
}
