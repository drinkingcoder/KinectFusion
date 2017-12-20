#include "Utility/LocalConfigurator.h"

int main() {
    auto configurator = new LocalConfigurator();
    Parameters params;

    configurator->ParseParametersFromJsonFile(params, "config.json");

    std::cout << "FileName: " << params.ConfigurationName;
    std::cout << std::endl;
    std::cout << "RGBFrameSuffix: " << params.RGBFrameSuffix;
    std::cout << std::endl;
    std::cout << "InputSize" << params.InputSize.x << "," << params.InputSize.y;
    std::cout << std::endl;
}
