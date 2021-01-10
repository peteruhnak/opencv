from typing import Dict, List, Optional, TypedDict
from jsgen import ClassInfo, JSWrapperGenerator, ArgInfo, FuncInfo, FuncVariant

from string import Template

base_types_template = Template(
    """
export type int = number
export type long = number
export type float = number
export type double = number
"""
)

const_template = Template(
    """
/** $value */
export const $js_name: number
"""
)

enum_template = Template(
    """
export enum $enum_name {
$enum_constants}
"""
)

enum_constant_template = Template(
    """  $const_name = $const_value,
"""
)

# Ignore these functions due to Embind limitations for now
ignore_list = [
    "locate",  # int&
    "minEnclosingCircle",  # float&
    "checkRange",
    "minMaxLoc",  # double*
    "floodFill",  # special case, implemented in core_bindings.cpp
    "phaseCorrelate",
    "randShuffle",
    "calibrationMatrixValues",  # double&
    "undistortPoints",  # global redefinition
    "CamShift",  # Rect&
    "meanShift",  # Rect&
]

arg_type_mapping = {
    "": "void",
    "cv::Mat": "Mat",
    "std::vector<int>": "IntVector|int[]",
    "std::vector<float>": "FloatVector|float[]",
    "vector_float": "FloatVector|float[]",
    "std::vector<double>": "DoubleVector|double[]",
    "std::vector<Point>": "PointVector",
    "std::vector<cv::Mat>": "MatVector",
    "std::vector<Rect>": "RectVector",
    "std::vector<KeyPoint>": "KeyPointVector",
    "std::vector<DMatch>": "DMatchVector",
    "std::vector<std::vector<DMatch>>": "DMatchVectorVector",
    "std::vector<char>": "unknown",
    "std::vector<uchar>": "unknown",
    "std::vector<std::vector<char>>": "unknown",
    "std::vector<std::vector<KeyPoint>>": "unknown",
    "std::vector<std::vector<Point>>": "unknown",
    "std::vector<String>": "unknown",
    "bool": "boolean",
    "size_t": "number",
    "String": "string",
    "Size": "SizeLike",
    "Point": "PointLike",
    "Point2f": "Point2fLike",
    "Rect": "RectLike",
    "RotatedRect": "RotatedRectLike",
    "Scalar": "ScalarLike",
    "TermCriteria": "TermCriteriaLike",
    "UsacParams": "unknown",
    "Moments": "MomentsLike",
    "Net": "unknown",
}

func_imports_template = Template(
    """import { int, float, double } from '../core/_types'
import { Mat } from '../core/Mat'
import { IntVector, FloatVector, PointVector, MatVector, RectVector, KeyPointVector, DMatchVector, DMatchVectorVector } from '../core/vectors'
import { DrawMatchesFlags } from './enums'
import { SizeLike, PointLike, Point2fLike, RectLike, TermCriteriaLike, ScalarLike, RotatedRectLike, MomentsLike } from '../core/valueObjects'
"""
)

func_template = Template(
    """
/**
$comment
 */
export function $func_name($args): $return_type
"""
)

method_template = Template(
    """
  /**
$comment
   */
  $func_name($args): $return_type
"""
)

constructor_template = Template(
    """
  /**
$comment
   */
  constructor($args)
"""
)

prop_template = Template(
    """
  $readonly$name: $type
"""
)

func_arg_template = Template("""$name$optional: $type""")

func_arg_doc_template = Template("@param $name  $comment")

classes_file_imports_template = Template(
    """import { int, float, double } from '../core/_types'
import { Mat } from '../core/Mat'
import { IntVector, FloatVector, DoubleVector, MatVector, RectVector, KeyPointVector, DMatchVector, DMatchVectorVector } from '../core/vectors'
import { AgastFeatureDetector_DetectorType, AKAZE_DescriptorType, DescriptorMatcher_MatcherType, FastFeatureDetector_DetectorType, HOGDescriptor_HistogramNormType, KAZE_DiffusivityType, ORB_ScoreType } from './enums'
import { SizeLike, PointLike, ScalarLike } from '../core/valueObjects'
import { EmClassHandle } from '../emscripten/emscripten'
"""
)

class_template = Template(
    """
/**
$comment
 */
export class $name extends $parent_name {
$body
}
"""
)


class TsGen:
    output_dir: str
    generator: JSWrapperGenerator
    modules_white_list: Dict[str, List[str]]

    def __init__(
        self, output_dir: str, generator: JSWrapperGenerator, modules_white_list: Dict[str, List[str]]
    ):
        self.output_dir = output_dir
        self.generator = generator
        self.bindings = []
        self.modules_white_list = modules_white_list

    def gen(self):
        self.gen_consts()
        self.gen_enums()
        self.gen_funcs()
        self.gen_classes()

    def is_module_ignored(self, module_name: str) -> bool:
        return module_name not in self.modules_white_list

    def is_method_ignored(self, method_name: str, module_name: str) -> bool:
        if method_name in ignore_list:
            return True
        return method_name not in self.modules_white_list[module_name]

    def gen_consts(self):
        bindings: List[str] = []
        # conflict between numerous CALIB_* constants from cv::fisheye::CALIB_* and from cv::CALIB_*
        added_consts: Dict[str, bool] = {}
        for ns_name, ns in sorted(self.generator.namespaces.items()):
            if ns_name.split(".")[0] != "cv":
                continue
            for name, const in sorted(ns.consts.items()):
                if name in added_consts:
                    continue
                added_consts[name] = True
                bindings.append(
                    const_template.substitute(js_name=name, value=const)
                )

        out_contents = "".join(bindings)
        self.write_file("constants", out_contents)

    def gen_funcs(self):
        bindings = ""
        for ns_name, ns in sorted(self.generator.namespaces.items()):
            if ns_name.split(".")[0] != "cv":
                continue
            for name, func in sorted(ns.funcs.items()):
                if self.is_method_ignored(name, ""):
                    continue

                ext_cnst = False
                # Check if the method is an external constructor
                for variant in func.variants:
                    if "Ptr<" in variant.rettype:
                        ext_cnst = True
                if ext_cnst:
                    continue

                # we are only generating types, so we dont need `with_wrapped_functions` (cf. embindgen.py)
                binding = self.gen_function(func, class_info=None)
                bindings += binding

        out_contents = ""
        out_contents += func_imports_template.substitute()
        out_contents += bindings
        self.write_file("functions", out_contents)
        pass

    def gen_function(self, func: FuncInfo, class_info: Optional[ClassInfo]):
        result = ""

        for var in func.variants:
            args = []
            for arg in var.args:
                arg_type = self.get_type(arg.tp)
                if var.name == 'inRange' and (arg.name == 'lowerb' or arg.name == 'upperb'):
                    arg_type = 'Mat | ScalarLike | number[]'

                args.append(
                    func_arg_template.substitute(
                        name=arg.name,
                        optional="?" if arg.defval != "" else "",
                        type=arg_type,
                    )
                )

            docstring = var.docstring
            if docstring is None or docstring == "":
                docstring = " "

            docstring = "\n".join(
                ["   * " + l.lstrip(" *") for l in docstring.splitlines()]
            )

            if class_info is None:
                result += func_template.substitute(
                    comment=docstring,  # " * " + "\n * ".join(doc_comment),
                    func_name=var.name,
                    args=", ".join(args),
                    return_type=self.get_type(var.rettype),
                )
            elif var.is_constructor:
                result += constructor_template.substitute(
                    comment=docstring,
                    args=", ".join(args),
                )
            elif 'Ptr<' in var.rettype:
                result += constructor_template.substitute(
                    comment=docstring,
                    args=", ".join(args),
                )
            else:
                result += method_template.substitute(
                    comment=docstring,
                    func_name=var.name,
                    args=", ".join(args),
                    return_type=self.get_type(var.rettype),
                )

        return result

    def get_type(self, type: str) -> str:
        type = type.replace("const ", "").rstrip("&")
        if type.startswith("Ptr<"):
            type = type[4:-1]
        if type in arg_type_mapping:
            type = arg_type_mapping[type]
        return type

    def gen_classes(self):
        class_exports = []
        for name, class_info in sorted(self.generator.classes.items()):
            if self.is_module_ignored(name):
                continue

            class_exports.append(self.gen_class(name, class_info))

        out_contents = ""
        out_contents += classes_file_imports_template.substitute()
        out_contents += "".join(class_exports)
        self.write_file("classes", out_contents)

    def gen_class(self, class_name, class_info):
        if class_name == "segmentation_IntelligentScissorsMB":
            class_name = "IntelligentScissorsMB"
        
        class_body = ""

        for prop in class_info.props:
            prop_body = prop_template.substitute(
                comment="   * " + prop.tp,
                name=prop.name,
                type=self.get_type(prop.tp),
                readonly="",  # apparently HOGDescriptor.nlevels is assigned to in `test_objdetect.ts`  #'readonly ' if prop.readonly else ''
            )
            class_body += prop_body

        for _, method in sorted(class_info.methods.items()):
            if method.cname in ignore_list:
                continue
            if not method.name in self.modules_white_list[method.class_name]:
                continue
            if method.is_constructor:
                pass
                method_body = self.gen_function(method, class_info)
                class_body += method_body
            else:
                method_body = self.gen_function(method, class_info)
                class_body += method_body

        extends = "EmClassHandle"
        if class_info.bases != []:
            if len(class_info.bases) > 1:
                print(f"WARNING: Class `${class_name}` has multiple parents: {class_info.bases}")
            parent_name = class_info.bases[0]
            # TODO: there are typing conflicts/overloads to be resolved for some parent classes
            if parent_name in ['Feature2D', 'DescriptorMatcher']:
              extends = parent_name

        docstring = class_info.docstring
        if docstring is None:
            docstring = ""
        docstring = "\n".join([" * " + l for l in docstring.splitlines()])

        return class_template.substitute(
            comment=docstring, name=class_name, parent_name=extends, body=class_body
        )

    def gen_enums(self):
        out_contents = ""

        for ns_name, ns in sorted(self.generator.namespaces.items()):
            if ns_name.split(".")[0] != "cv":
                continue
            for name, enum in sorted(ns.enums.items()):
                enum_name = name.replace("cv.", "").replace(".", "_")
                if "unnamed" in enum_name:
                    continue  # unnamed seems like an internal thing
                if enum_name == "_OutputArray_DepthMask":
                    continue  # ignored, because the dependent constants are only in core_bindings.cpp
                enum_constants = ""
                for const in enum:
                    const_name = const[0].replace("const ", "")
                    last_dot_pos = const_name.rfind(".") + 1
                    const_name = const_name[last_dot_pos:]
                    const_value = const[1]
                    enum_constants += enum_constant_template.substitute(
                        const_name=const_name, const_value=const_value
                    )
                out_contents += enum_template.substitute(
                    enum_name=enum_name, enum_constants=enum_constants
                )

        self.write_file("enums", out_contents)

    def write_file(self, base_name: str, contents: str):
        file_path = self.output_dir + "/" + base_name + ".d.ts"
        with open(file_path, "w", newline='\n') as f:
            f.write(contents)
        print(f'Written {file_path}')
