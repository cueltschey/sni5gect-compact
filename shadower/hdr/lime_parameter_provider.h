#ifndef LIME_SDR_SOURCE_H
#define LIME_SDR_SOURCE_H
#include "limesuiteng/LimePlugin.h"
#include "limesuiteng/limesuiteng.hpp"
#include <string>
class LimeParamProvider : public LimeSettingsProvider
{
private:
  static std::string trim(const std::string& s)
  {
    std::string out = s;
    while (!out.empty() && std::isspace(out[0]))
      out = out.substr(1);
    while (!out.empty() && std::isspace(out[out.size() - 1]))
      out = out.substr(0, out.size() - 1);
    return out;
  }

  void argsToMap(const std::string& args)
  {
    bool        inKey = true;
    std::string key, val;
    for (size_t i = 0; i < args.size(); i++) {
      const char ch = args[i];
      if (inKey) {
        if (ch == ':')
          inKey = false;
        else if (ch == ',')
          inKey = true;
        else
          key += ch;
      } else {
        if (ch == ',')
          inKey = true;
        else
          val += ch;
      }
      if ((inKey && !val.empty()) || ((i + 1) == args.size())) {
        key = trim(key);
        val = trim(val);
        printf("Key:Value{ %s:%s }\n", key.c_str(), val.c_str());
        if (!key.empty()) {
          if (val[0] == '"')
            strings[key] = val.substr(1, val.size() - 2);
          else
            numbers[key] = stod(val);
        }
        key = "";
        val = "";
      }
    }
  }

public:
  LimeParamProvider(const char* args) : mArgs(args) { argsToMap(mArgs); }

  bool GetString(std::string& dest, const char* varname) override
  {
    auto iter = strings.find(std::string(varname));
    if (iter == strings.end())
      return false;

    printf("provided: %s\n", varname);

    dest = iter->second;
    return true;
  }

  bool GetDouble(double& dest, const char* varname) override
  {
    auto iter = numbers.find(varname);
    if (iter == numbers.end())
      return false;

    dest = iter->second;
    return true;
  }

private:
  std::string                                  mArgs;
  std::unordered_map<std::string, double>      numbers;
  std::unordered_map<std::string, std::string> strings;
};

#endif // LIME_SDR_SOURCE_H