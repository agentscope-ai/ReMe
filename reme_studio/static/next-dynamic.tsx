import {
  lazy,
  Suspense,
  type ComponentType,
  type LazyExoticComponent,
} from "react";

type DynamicOptions = {
  ssr?: boolean;
};

export default function dynamic<Props extends object>(
  loader: () => Promise<{ default: ComponentType<Props> }>,
  options?: DynamicOptions,
): ComponentType<Props> {
  void options;
  const LazyComponent: LazyExoticComponent<ComponentType<Props>> = lazy(loader);

  return function DynamicComponent(props: Props) {
    return (
      <Suspense fallback={null}>
        <LazyComponent {...props} />
      </Suspense>
    );
  };
}
